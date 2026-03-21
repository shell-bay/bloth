"""Bloth RMSNorm nn.Module wrapper."""
import torch
import torch.nn as nn
from ..kernels.rms_norm import rms_norm as _rms_norm_fn, bloth_rms_norm


class BlothRMSNorm(nn.Module):
    """Drop-in for LlamaRMSNorm / MistralRMSNorm / QwenRMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm_fn(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _rms_norm_fn(x, weight, eps)
