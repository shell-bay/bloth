"""
Bloth Kernels Package
FIX: all names exported here MUST match exactly what model_patcher.py imports.
bloth_rms_norm is the alias expected by model_patcher — do NOT rename.
"""

from .rms_norm            import bloth_rms_norm, rms_norm
from .fused_norm_qkv_rope import fused_norm_rope
from .flash_attention     import flash_attention, flash_attention_varlen
from .cross_entropy       import fused_cross_entropy, fused_linear_cross_entropy
from .adaptive_fp8        import AdaptiveDelayedScaler, get_scaler, quantize_tensor_fp8

__all__ = [
    "bloth_rms_norm",
    "rms_norm",
    "fused_norm_rope",
    "flash_attention",
    "flash_attention_varlen",
    "fused_cross_entropy",
    "fused_linear_cross_entropy",
    "AdaptiveDelayedScaler",
    "get_scaler",
    "quantize_tensor_fp8",
]
