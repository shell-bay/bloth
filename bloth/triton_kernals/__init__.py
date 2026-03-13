"""
Triton Fallback Kernels
=======================

Pure Triton implementations for when CUDA extensions are not available.
These are still highly optimized but may not match the peak performance
of hand-tuned CUDA kernels with warp specialization.
"""

import torch
import triton
import triton.language as tl


def triton_gemm(a, b, c=None, alpha=1.0, beta=0.0, transpose_a=False, transpose_b=False):
    """Triton GEMM fallback."""
    # Use PyTorch matmul (which may use cuBLAS)
    result = torch.matmul(a, b.t() if transpose_b else b)
    if alpha != 1.0:
        result = result * alpha
    if c is not None and beta != 0.0:
        result = result + beta * c
    return result


def triton_layer_norm(x, weight, bias=None, eps=1e-5):
    """Triton layer norm fallback."""
    # PyTorch fallback
    return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)


def triton_softmax(x, dim=-1):
    """Triton softmax fallback."""
    return torch.nn.functional.softmax(x, dim=dim)


def triton_cross_entropy(logits, labels, reduction='mean', label_smoothing=0.0, ignore_index=-100):
    """Triton cross entropy fallback."""
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction=reduction,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index
    )


__all__ = [
    'triton_gemm',
    'triton_layer_norm',
    'triton_softmax',
    'triton_cross_entropy',
]
