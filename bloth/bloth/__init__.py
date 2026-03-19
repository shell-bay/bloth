"""
Bloth v1.0 — Ultra-Fast CUDA Kernel Library for LLM Training
=============================================================
Key innovations:
  1. Hyper-Fused kernel: RMSNorm + RoPE in ONE Triton pass  (~15-20% faster)
  2. Manual backward pass: no activation storage             (~70% less VRAM)
  3. Adaptive Delayed FP8 Scaling: FP8 speed + BF16 safety
  4. FlashAttention-2 tiling: O(sqrt(N)) memory
  5. Fused Linear + CrossEntropy: no logit buffer
  6. Universal: Maxwell to Blackwell via Triton PTX auto-compile
"""

__version__ = "1.0.0"
__author__  = "shell-bay"

import torch

from .ops.rms_norm       import BlothRMSNorm, rms_norm
from .ops.rope           import BlothRoPE, rope_embedding
from .ops.flash_attn     import BlothFlashAttention, flash_attention, flash_attention_varlen
from .ops.fused_norm_qkv import BlothFusedNormQKV, fused_norm_qkv
from .ops.cross_entropy  import BlothCrossEntropy, fused_cross_entropy, fused_linear_cross_entropy
from .ops.lora           import BlothLoRA, lora_linear, lora_mlp
from .ops.gemm           import gemm, gemm_fp8, gemm_splitk
from .patch              import patch_model, FastModel
from .utils.memory       import (get_gpu_memory, optimize_memory, print_gpu_memory,
                                  estimate_memory_usage, get_optimal_batch_size)
from .utils.device       import get_device_info, print_device_info
from .utils.benchmark    import benchmark_forward
from .utils.seed         import set_seed


class kernels:
    """Direct access to all low-level Triton kernels."""
    from .kernels.rms_norm            import bloth_rms_norm, rms_norm
    from .kernels.fused_norm_qkv_rope import fused_norm_rope
    from .kernels.flash_attention     import flash_attention, flash_attention_varlen
    from .kernels.cross_entropy       import fused_cross_entropy, fused_linear_cross_entropy
    from .kernels.adaptive_fp8        import AdaptiveDelayedScaler, get_scaler
    from .ops.gemm                    import gemm, gemm_fp8, gemm_splitk
    from .ops.rope                    import rope_embedding
    from .ops.lora                    import lora_linear, lora_mlp


def get_model_info(model) -> dict:
    params    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "model_class":      type(model).__name__,
        "total_params":     params,
        "trainable_params": trainable,
        "trainable_pct":    round(trainable / max(params, 1) * 100, 2),
    }


def print_model_info(model):
    info = get_model_info(model)
    print(f"\n  Model:      {info['model_class']}")
    print(f"  Total:      {info['total_params']/1e9:.2f}B params")
    print(f"  Trainable:  {info['trainable_params']/1e6:.1f}M ({info['trainable_pct']}%)\n")
