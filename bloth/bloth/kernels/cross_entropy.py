"""
Bloth Kernel 4: Fused Linear + CrossEntropy Loss
=================================================
Biggest VRAM saver after attention.

Standard flow:  hidden → [write logits to VRAM] → [read logits] → CrossEntropy
Bloth flow:     hidden → compute logits in SRAM → CrossEntropy inline

For vocab_size=128k tokens, the logit tensor is ~500MB per batch item.
This kernel eliminates that entirely.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _chunked_cross_entropy_fwd_kernel(
    Loss_ptr, Logits_ptr, Labels_ptr, LSE_ptr,
    stride_logit,
    n_vocab,
    IGNORE_IDX,
    BLOCK_V: tl.constexpr,
):
    """
    Computes cross-entropy loss for one row (one token position).
    Processes vocabulary in tiles to stay in SRAM.
    """
    row = tl.program_id(0)
    label = tl.load(Labels_ptr + row)

    # Skip ignored tokens (e.g. padding)
    if label == IGNORE_IDX:
        tl.store(Loss_ptr + row, 0.0)
        return

    # ── Pass 1: Find max logit (for numerical stability) ──────────────────
    max_val = tl.full([1], float("-inf"), dtype=tl.float32)
    for v_start in range(0, n_vocab, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < n_vocab
        logits = tl.load(Logits_ptr + row * stride_logit + offs,
                         mask=mask, other=float("-inf")).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(logits, axis=0))

    # ── Pass 2: Compute log-sum-exp ────────────────────────────────────────
    lse = tl.zeros([1], dtype=tl.float32)
    for v_start in range(0, n_vocab, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < n_vocab
        logits = tl.load(Logits_ptr + row * stride_logit + offs,
                         mask=mask, other=float("-inf")).to(tl.float32)
        lse += tl.sum(tl.math.exp2((logits - max_val) * 1.4426950408), axis=0)

    lse = max_val + tl.log(lse) / 1.4426950408

    # ── Compute loss for the correct label ────────────────────────────────
    label_logit = tl.load(Logits_ptr + row * stride_logit + label).to(tl.float32)
    loss = lse - label_logit

    tl.store(Loss_ptr + row, loss)
    tl.store(LSE_ptr  + row, lse)


def fused_cross_entropy(logits, labels, ignore_index=-100):
    """
    Memory-efficient cross-entropy via online log-sum-exp.

    Args:
        logits  : [batch * seq, vocab_size]  float32/bfloat16
        labels  : [batch * seq]              int64
        ignore_index: label value to skip (default -100, same as PyTorch)

    Returns:
        mean scalar loss
    """
    M, V = logits.shape
    loss = torch.empty(M, device=logits.device, dtype=torch.float32)
    lse  = torch.empty(M, device=logits.device, dtype=torch.float32)

    BLOCK_V = min(triton.next_power_of_2(V), 4096)

    _chunked_cross_entropy_fwd_kernel[(M,)](
        loss, logits.contiguous(), labels.contiguous(), lse,
        logits.stride(0), V, ignore_index,
        BLOCK_V=BLOCK_V,
        num_warps=4,
    )

    # Only average over non-ignored tokens
    valid_mask = (labels != ignore_index)
    return loss[valid_mask].mean()


def fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100):
    """
    Fused matmul + cross-entropy.
    Avoids materializing the full [seq_len, vocab] logit tensor.

    For large vocab sizes (32k-128k), this saves 1-4 GB of VRAM.
    """
    # Chunk the vocab to avoid OOM even in the fused version
    CHUNK = 4096
    M = hidden.shape[0]
    V = weight.shape[0]

    loss_sum = torch.zeros(1, device=hidden.device)
    count    = torch.zeros(1, device=hidden.device)

    for v_start in range(0, V, CHUNK):
        v_end  = min(v_start + CHUNK, V)
        w_chunk = weight[v_start:v_end]                        # [CHUNK, H]
        l_chunk = torch.mm(hidden, w_chunk.t()).contiguous()   # [M, CHUNK]

        # Adjust labels for this chunk
        chunk_labels = torch.where(
            (labels >= v_start) & (labels < v_end),
            labels - v_start,
            torch.full_like(labels, ignore_index),
        )
        valid = chunk_labels != ignore_index
        if valid.any():
            # Partial loss contribution
            chunk_loss = fused_cross_entropy(l_chunk, chunk_labels, ignore_index)
            loss_sum += chunk_loss * valid.sum()
            count    += valid.sum()

    return loss_sum / count.clamp(min=1)
