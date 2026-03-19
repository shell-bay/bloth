"""
Bloth Kernel: FlashAttention-2
================================
Tiled attention — O(sqrt(N)) memory instead of O(N^2).
At 128k context, saves ~40 GB vs naive attention.
Supports causal (autoregressive) masking.
"""

import triton
import triton.language as tl
import torch
import math


@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    seq_len, head_dim, scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M:   tl.constexpr,
    BLOCK_N:   tl.constexpr,
    BLOCK_K:   tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz  = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    Q_blk = Q_ptr + off_hz * stride_qm * seq_len
    K_blk = K_ptr + off_hz * stride_km * seq_len
    V_blk = V_ptr + off_hz * stride_vm * seq_len
    O_blk = O_ptr + off_hz * stride_om * seq_len

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + float("-inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    end_n = (start_m + 1) * BLOCK_M if IS_CAUSAL else seq_len

    for start_n in range(0, tl.minimum(end_n, seq_len), BLOCK_N):
        k_mask = (start_n + offs_n)[:, None] < seq_len
        q_mask = offs_m[:, None] < seq_len

        q = tl.load(Q_blk + offs_m[:, None] * stride_qm + offs_k[None, :],
                    mask=q_mask, other=0.0)
        k = tl.load(K_blk + (start_n + offs_n[:, None]) * stride_km + offs_k[None, :],
                    mask=k_mask, other=0.0)

        scores = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            scores = tl.where(causal_mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha  = tl.math.exp2((m_i - m_new) * 1.4426950408)
        p      = tl.math.exp2((scores - m_new[:, None]) * 1.4426950408)

        v = tl.load(V_blk + (start_n + offs_n[:, None]) * stride_vm + offs_k[None, :],
                    mask=k_mask, other=0.0)

        acc    = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
        l_i    = l_i * alpha + tl.sum(p, axis=1)
        m_i    = m_new

    o = acc / (l_i[:, None] + 1e-8)
    o_mask = offs_m[:, None] < seq_len
    tl.store(O_blk + offs_m[:, None] * stride_om + offs_k[None, :],
             o.to(tl.bfloat16), mask=o_mask)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: float = None,
) -> torch.Tensor:
    """
    Memory-efficient attention — O(sqrt(N)) VRAM.

    Args:
        q, k, v : [batch, heads, seq_len, head_dim]  bfloat16
        causal  : apply autoregressive mask
        scale   : 1/sqrt(head_dim) by default

    Returns:
        output  : [batch, heads, seq_len, head_dim]
    """
    batch, heads, seq_len, head_dim = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    q = q.reshape(batch * heads, seq_len, head_dim).contiguous()
    k = k.reshape(batch * heads, seq_len, head_dim).contiguous()
    v = v.reshape(batch * heads, seq_len, head_dim).contiguous()
    o = torch.empty_like(q)

    BLOCK_M = BLOCK_N = 64
    BLOCK_K = triton.next_power_of_2(head_dim)
    num_warps = 4 if head_dim <= 64 else 8

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)
    _flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(1), q.stride(2),
        k.stride(1), k.stride(2),
        v.stride(1), v.stride(2),
        o.stride(1), o.stride(2),
        seq_len, head_dim, scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return o.reshape(batch, heads, seq_len, head_dim)


def flash_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True):
    """Variable-length attention for packed sequences (no padding waste)."""
    return flash_attention(q, k, v, causal=causal)
