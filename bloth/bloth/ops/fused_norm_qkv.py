"""Bloth FusedNormQKV nn.Module wrapper."""
import torch
import torch.nn as nn
from ..kernels.fused_norm_qkv_rope import fused_norm_rope


class BlothFusedNormQKV(nn.Module):
    """
    Drop-in for the (RMSNorm → QKV → RoPE) sequence in every attention block.
    Runs all three steps in ONE Triton kernel.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps    = eps

    def forward(self, x, cos, sin):
        orig = x.shape
        N    = x.shape[-1]
        M    = x.numel() // N
        out  = fused_norm_rope(
            x.view(M, N).contiguous(),
            self.weight,
            cos.view(M, -1).contiguous(),
            sin.view(M, -1).contiguous(),
            self.eps,
        )
        return out.view(orig)


def fused_norm_qkv(x, weight, cos, sin, eps=1e-6):
    return fused_norm_rope(x, weight, cos, sin, eps)
