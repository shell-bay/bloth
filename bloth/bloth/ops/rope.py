"""
Bloth RoPE op — Rotary Positional Embedding nn.Module + functional API.
Supports LLaMA, Mistral, Qwen, Gemma rotary styles.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def _rope_fwd_kernel(
    Out_ptr, X_ptr, Cos_ptr, Sin_ptr,
    stride_x, stride_cos,
    seq_len, head_dim,
    BLOCK: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    half = head_dim // 2

    # Load first half and second half of the head
    mask1 = offs < half
    mask2 = (offs >= half) & (offs < head_dim)

    x1 = tl.load(X_ptr + row * stride_x + offs,        mask=mask1, other=0.0).to(tl.float32)
    x2 = tl.load(X_ptr + row * stride_x + offs + half, mask=mask1, other=0.0).to(tl.float32)

    cos_ = tl.load(Cos_ptr + (row % seq_len) * stride_cos + offs, mask=mask1, other=1.0).to(tl.float32)
    sin_ = tl.load(Sin_ptr + (row % seq_len) * stride_cos + offs, mask=mask1, other=0.0).to(tl.float32)

    y1 = x1 * cos_ - x2 * sin_
    y2 = x1 * sin_ + x2 * cos_

    tl.store(Out_ptr + row * stride_x + offs,        y1.to(tl.bfloat16), mask=mask1)
    tl.store(Out_ptr + row * stride_x + offs + half, y2.to(tl.bfloat16), mask=mask1)


def rope_embedding(x, cos, sin, position_ids=None):
    """
    Apply rotary positional embeddings to x.
    x   : [batch, heads, seq, head_dim]
    cos : [seq, head_dim//2]
    sin : [seq, head_dim//2]
    """
    b, h, s, d = x.shape
    x_flat = x.reshape(b * h * s, d).contiguous()
    out    = torch.empty_like(x_flat)

    # Expand cos/sin to match batch*heads
    cos_exp = cos.unsqueeze(0).expand(b * h, -1, -1).reshape(b * h * s, d // 2).contiguous()
    sin_exp = sin.unsqueeze(0).expand(b * h, -1, -1).reshape(b * h * s, d // 2).contiguous()

    BLOCK = triton.next_power_of_2(d)
    _rope_fwd_kernel[(b * h * s,)](
        out, x_flat, cos_exp, sin_exp,
        x_flat.stride(0), cos_exp.stride(0),
        s, d, BLOCK=BLOCK,
        num_warps=4,
    )
    return out.reshape(b, h, s, d)


class BlothRoPE(nn.Module):
    """Pre-computes and caches cos/sin tables for the configured max sequence length."""

    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        self.base        = base
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, dtype=torch.float32) / half))
        t   = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                   # [seq, half]
        self.register_buffer("cos_cache", freqs.cos().bfloat16(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin().bfloat16(), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len and seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return rope_embedding(x, self.cos_cache, self.sin_cache)
