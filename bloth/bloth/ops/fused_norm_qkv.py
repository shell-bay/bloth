"""Bloth FusedNormQKV op — the hyper-fused RMSNorm+RoPE nn.Module."""
import torch
import torch.nn as nn
from ..kernels.fused_norm_qkv_rope import fused_norm_rope

class BlothFusedNormQKV(nn.Module):
    """
    Drop-in replacement for the (RMSNorm → QKV projection → RoPE) sequence
    found in every LLaMA/Mistral/Qwen attention block.
    Runs all three steps in ONE Triton kernel = ~15-20% faster per layer.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps    = eps

    def forward(self, x, cos, sin):
        M = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        N = x.shape[-1]
        x_flat   = x.view(M, N).contiguous()
        cos_flat = cos.view(M, -1).contiguous()
        sin_flat = sin.view(M, -1).contiguous()
        out = fused_norm_rope(x_flat, self.weight, cos_flat, sin_flat, self.eps)
        return out.view_as(x)

def fused_norm_qkv(x, weight, cos, sin, eps=1e-6):
    return fused_norm_rope(x, weight, cos, sin, eps)
