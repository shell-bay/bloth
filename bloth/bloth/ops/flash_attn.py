"""Bloth FlashAttention nn.Module wrapper."""
import torch.nn as nn
from ..kernels.flash_attention import flash_attention as _fa, flash_attention_varlen as _fa_vl


class BlothFlashAttention(nn.Module):
    def __init__(self, causal: bool = True):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v):
        return _fa(q, k, v, causal=self.causal)


def flash_attention(q, k, v, causal=True):
    return _fa(q, k, v, causal=causal)


def flash_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True):
    return _fa_vl(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)
