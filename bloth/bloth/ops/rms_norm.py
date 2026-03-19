"""
Bloth RMSNorm op — nn.Module wrapper around the Triton kernel.
Drop-in replacement for LlamaRMSNorm, MistralRMSNorm, etc.
"""
import torch
import torch.nn as nn
from ..kernels.rms_norm import rms_norm as _rms_norm_fn

class BlothRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps    = eps

    def forward(self, x):
        return _rms_norm_fn(x, self.weight, self.eps)

    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"

# Functional alias
def rms_norm(x, weight, eps=1e-6):
    return _rms_norm_fn(x, weight, eps)
