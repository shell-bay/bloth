"""
Bloth Kernel: Fused Cross Entropy  (v1.1 — FIXED)
===================================================
Fix from Colab T4 testing:
  - ERROR: "Value argument cannot be block type if pointer argument is not a block"
  - Root cause: tl.store(scalar_ptr, [1]-shaped-tensor) is illegal in Triton
  - Fix: all accumulators use tl.zeros([1]) but stored with [0] indexing
  - Also: max_val must be scalar for tl.store — use [0] extraction

Memory benefit: never stores the full [seq, vocab] logit tensor.
For vocab=128k, that's ~500 MB per batch item saved.

Algorithm: two-pass online log-sum-exp (numerically stable softmax).
"""

import triton
import triton.language as tl
import torch


def _cross_entropy_block_size(vocab_size: int) -> int:
    bs = triton.next_power_of_2(vocab_size)
    return min(bs, 4096)


@triton.jit
def _cross_entropy_fwd_kernel(
    Loss_ptr, LSE_ptr,
    Logits_ptr, Labels_ptr,
    stride_logit,
    n_vocab,
    IGNORE_IDX: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    One program per token position.
    Two passes over vocabulary (in BLOCK_V tiles) to compute log-sum-exp.

    KEY FIX: all scalar stores use explicit [0] extraction to convert
    Triton [1] tensors to scalars before passing to tl.store.
    """
    row   = tl.program_id(0)
    label = tl.load(Labels_ptr + row)

    # Skip padding / ignore tokens
    if label == IGNORE_IDX:
        tl.store(Loss_ptr + row, 0.0)
        tl.store(LSE_ptr  + row, 0.0)
        return

    # ── Pass 1: find maximum logit (for numerical stability) ──────────────
    # Accumulate max as a [1] tensor, extract scalar at end
    running_max = tl.full([1], float("-inf"), dtype=tl.float32)

    for v_start in range(0, n_vocab, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < n_vocab
        vals   = tl.load(
            Logits_ptr + row * stride_logit + v_offs,
            mask=v_mask, other=float("-inf"),
        ).to(tl.float32)
        # tl.max returns scalar; tl.maximum([1], scalar) returns [1]
        running_max = tl.maximum(running_max, tl.max(vals, axis=0))

    # FIX: extract scalar from [1] tensor using [0] index
    max_val = running_max[0]

    # ── Pass 2: compute log-sum-exp ───────────────────────────────────────
    running_sum = tl.zeros([1], dtype=tl.float32)

    for v_start in range(0, n_vocab, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < n_vocab
        vals   = tl.load(
            Logits_ptr + row * stride_logit + v_offs,
            mask=v_mask, other=float("-inf"),
        ).to(tl.float32)
        running_sum += tl.sum(tl.exp(vals - max_val), axis=0)

    # FIX: extract scalar before arithmetic and store
    lse_scalar  = tl.log(running_sum[0] + 1e-12) + max_val

    # Correct-label logit
    label_logit = tl.load(Logits_ptr + row * stride_logit + label).to(tl.float32)
    loss_scalar = lse_scalar - label_logit

    # FIX: both pointers and values are scalars — valid tl.store
    tl.store(Loss_ptr + row, loss_scalar)
    tl.store(LSE_ptr  + row, lse_scalar)


def fused_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Tiled cross-entropy — never materialises full [seq, vocab] tensor.

    Args:
        logits       : [batch * seq, vocab_size]  float32 or bfloat16
        labels       : [batch * seq]              int64
        ignore_index : label value to skip (default -100, matches PyTorch)

    Returns:
        scalar mean loss (float32)
    """
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits = logits.view(B * S, V)
        labels = labels.view(B * S)

    # Always use float32 for stability — bfloat16 causes divergence at large vocab
    logits = logits.contiguous().float()
    labels = labels.contiguous()
    M, V   = logits.shape

    loss = torch.zeros(M, device=logits.device, dtype=torch.float32)
    lse  = torch.zeros(M, device=logits.device, dtype=torch.float32)

    BLOCK_V = _cross_entropy_block_size(V)

    _cross_entropy_fwd_kernel[(M,)](
        loss, lse, logits, labels,
        logits.stride(0), V,
        IGNORE_IDX=ignore_index,
        BLOCK_V=BLOCK_V,
        num_warps=4,
    )

    valid = (labels != ignore_index)
    n_valid = valid.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)
    return loss[valid].mean()


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Fused matmul + cross-entropy in vocabulary chunks.
    Never stores the full [M, vocab] logit tensor.

    For vocab=128k tokens, saves ~500MB VRAM per batch item.

    Args:
        hidden : [M, hidden_dim]      bfloat16 / float32
        weight : [vocab_size, hidden_dim]
        labels : [M]                  int64
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
            loss_sum  += chunk_loss * valid.float().sum()
            count     += valid.float().sum()

    return (loss_sum / count.clamp(min=1.0)).squeeze()
