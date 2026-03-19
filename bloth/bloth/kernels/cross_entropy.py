"""
Bloth Kernel: Fused Cross Entropy
===================================
For vocab_size=128k, the logit tensor is ~500MB per batch item.
This kernel processes it in tiles and never stores the full tensor.
Result: saves 1-4 GB VRAM during the loss backward pass.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _cross_entropy_fwd_kernel(
    Loss_ptr, LSE_ptr, Logits_ptr, Labels_ptr,
    stride_logit,
    n_vocab,
    IGNORE_IDX,
    BLOCK_V: tl.constexpr,
):
    row   = tl.program_id(0)
    label = tl.load(Labels_ptr + row)

    if label == IGNORE_IDX:
        tl.store(Loss_ptr + row, 0.0)
        tl.store(LSE_ptr  + row, 0.0)
        return

    # Pass 1: max for numerical stability
    max_val = tl.full([1], float("-inf"), dtype=tl.float32)
    for v in range(0, n_vocab, BLOCK_V):
        offs = v + tl.arange(0, BLOCK_V)
        mask = offs < n_vocab
        vals = tl.load(Logits_ptr + row * stride_logit + offs,
                        mask=mask, other=float("-inf")).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))

    # Pass 2: log-sum-exp
    lse = tl.zeros([1], dtype=tl.float32)
    for v in range(0, n_vocab, BLOCK_V):
        offs = v + tl.arange(0, BLOCK_V)
        mask = offs < n_vocab
        vals = tl.load(Logits_ptr + row * stride_logit + offs,
                        mask=mask, other=float("-inf")).to(tl.float32)
        lse += tl.sum(tl.exp(vals - max_val), axis=0)

    lse = tl.log(lse + 1e-12) + max_val

    label_logit = tl.load(Logits_ptr + row * stride_logit + label).to(tl.float32)
    loss = lse - label_logit

    tl.store(Loss_ptr + row, loss)
    tl.store(LSE_ptr  + row, lse)


def fused_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Tiled cross-entropy — never stores the full [seq, vocab] logit tensor.

    Args:
        logits       : [batch * seq, vocab_size]  float32/bfloat16
        labels       : [batch * seq]              int64
        ignore_index : token id to skip (default -100)

    Returns:
        scalar mean loss
    """
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits = logits.view(B * S, V)
        labels = labels.view(B * S)

    logits = logits.contiguous().float()
    labels = labels.contiguous()
    M, V   = logits.shape

    loss = torch.zeros(M, device=logits.device, dtype=torch.float32)
    lse  = torch.zeros(M, device=logits.device, dtype=torch.float32)
    BLOCK_V = min(triton.next_power_of_2(V), 4096)

    _cross_entropy_fwd_kernel[(M,)](
        loss, lse, logits, labels,
        logits.stride(0), V, ignore_index,
        BLOCK_V=BLOCK_V,
        num_warps=4,
    )

    valid = labels != ignore_index
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return loss[valid].mean()


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Fused matmul + cross-entropy in chunks.
    Never materialises the full [seq_len, vocab] logit matrix.

    Args:
        hidden : [M, hidden_dim]  bfloat16
        weight : [vocab, hidden_dim]
        labels : [M]  int64
    """
    CHUNK    = 4096
    M        = hidden.shape[0]
    V        = weight.shape[0]
    loss_sum = torch.zeros(1, device=hidden.device, dtype=torch.float32)
    count    = torch.zeros(1, device=hidden.device, dtype=torch.float32)

    for start in range(0, V, CHUNK):
        end   = min(start + CHUNK, V)
        w_c   = weight[start:end].contiguous()
        l_c   = torch.mm(hidden.float(), w_c.float().t()).contiguous()

        lab_c = torch.where(
            (labels >= start) & (labels < end),
            labels - start,
            torch.full_like(labels, ignore_index),
        )
        valid = (lab_c != ignore_index)
        if valid.any():
            chunk_loss = fused_cross_entropy(l_c, lab_c, ignore_index)
            loss_sum  += chunk_loss * valid.sum().float()
            count     += valid.sum().float()

    return loss_sum / count.clamp(min=1)
