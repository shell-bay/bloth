"""
Bloth GPU Auto-Configuration
==============================
Detects your GPU architecture and enables the right optimizations.
No manual configuration needed - Bloth picks the best settings for you.

Supported architectures:
  SM60/61  - Pascal     (GTX 10-series)           → Basic Triton
  SM70/72  - Volta      (V100, Titan V)            → Tensor Cores
  SM75     - Turing     (RTX 20-series, T4)        → INT8 + Tensor Cores
  SM80/86  - Ampere     (RTX 30-series, A100)      → BF16 + TF32 + Async copies
  SM89     - Ada        (RTX 40-series, L4, L40)   → FP8 capable
  SM90     - Hopper     (H100, H200)               → FP8 + TMA + Warp Groups
  SM100    - Blackwell  (B100, B200, GB200)        → FP4 + NVLink5 + TMA2
"""

import torch
from dataclasses import dataclass, field
from typing import Optional
import warnings


@dataclass
class GPUCapabilities:
    """Describes what a GPU can do for Bloth's kernels."""
    name: str = "Unknown GPU"
    sm_major: int = 0
    sm_minor: int = 0
    total_memory_gb: float = 0.0
    num_sms: int = 0

    # Feature flags
    has_tensor_cores: bool = False
    has_bf16: bool = False
    has_tf32: bool = False
    has_async_copy: bool = False
    has_fp8: bool = False
    has_tma: bool = False
    has_warp_specialization: bool = False
    has_fp4: bool = False

    # Performance settings (auto-tuned by architecture)
    recommended_dtype: torch.dtype = torch.float16
    recommended_block_size: int = 128
    max_num_warps: int = 8
    l2_cache_mb: float = 0.0

    @property
    def compute_capability(self) -> str:
        return f"SM{self.sm_major}{self.sm_minor}"

    @property
    def is_hopper_or_newer(self) -> bool:
        return self.sm_major >= 9

    @property
    def is_ampere_or_newer(self) -> bool:
        return self.sm_major >= 8

    @property
    def is_turing_or_newer(self) -> bool:
        return (self.sm_major > 7) or (self.sm_major == 7 and self.sm_minor >= 5)


def detect_gpu() -> GPUCapabilities:
    """
    Automatically detect GPU and return its capabilities.
    Returns a default CPU-friendly config if no GPU found.
    """
    if not torch.cuda.is_available():
        warnings.warn(
            "No CUDA GPU detected. Bloth requires an NVIDIA GPU. "
            "Falling back to PyTorch defaults.",
            UserWarning,
            stacklevel=2
        )
        return GPUCapabilities(name="CPU (No GPU)")

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    caps = GPUCapabilities(
        name=name,
        sm_major=major,
        sm_minor=minor,
        total_memory_gb=total_mem,
        num_sms=num_sms,
    )

    # ── SM60/61: Pascal (GTX 1060, 1070, 1080, 1080 Ti, P100) ──
    if major == 6:
        caps.has_tensor_cores = False
        caps.has_bf16 = False
        caps.recommended_dtype = torch.float16
        caps.recommended_block_size = 64
        caps.max_num_warps = 4
        caps.l2_cache_mb = 2.0

    # ── SM70/72: Volta (V100, Titan V) ──
    elif major == 7 and minor < 5:
        caps.has_tensor_cores = True
        caps.has_bf16 = False
        caps.recommended_dtype = torch.float16
        caps.recommended_block_size = 128
        caps.max_num_warps = 8
        caps.l2_cache_mb = 6.0

    # ── SM75: Turing (RTX 2080, T4) ──
    elif major == 7 and minor == 5:
        caps.has_tensor_cores = True
        caps.has_bf16 = False
        caps.recommended_dtype = torch.float16
        caps.recommended_block_size = 128
        caps.max_num_warps = 8
        caps.l2_cache_mb = 6.0

    # ── SM80/86: Ampere (RTX 3090, A100, A10) ──
    elif major == 8 and minor <= 6:
        caps.has_tensor_cores = True
        caps.has_bf16 = True
        caps.has_tf32 = True
        caps.has_async_copy = True
        caps.recommended_dtype = torch.bfloat16
        caps.recommended_block_size = 128
        caps.max_num_warps = 8
        caps.l2_cache_mb = 40.0 if "A100" in name else 6.0

    # ── SM89: Ada Lovelace (RTX 4090, L4, L40S) ──
    elif major == 8 and minor == 9:
        caps.has_tensor_cores = True
        caps.has_bf16 = True
        caps.has_tf32 = True
        caps.has_async_copy = True
        caps.has_fp8 = True
        caps.recommended_dtype = torch.bfloat16
        caps.recommended_block_size = 256
        caps.max_num_warps = 16
        caps.l2_cache_mb = 72.0

    # ── SM90: Hopper (H100, H200) ──
    elif major == 9:
        caps.has_tensor_cores = True
        caps.has_bf16 = True
        caps.has_tf32 = True
        caps.has_async_copy = True
        caps.has_fp8 = True
        caps.has_tma = True
        caps.has_warp_specialization = True
        caps.recommended_dtype = torch.bfloat16
        caps.recommended_block_size = 256
        caps.max_num_warps = 16
        caps.l2_cache_mb = 50.0

    # ── SM100: Blackwell (B100, B200, GB200) ──
    elif major >= 10:
        caps.has_tensor_cores = True
        caps.has_bf16 = True
        caps.has_tf32 = True
        caps.has_async_copy = True
        caps.has_fp8 = True
        caps.has_fp4 = True
        caps.has_tma = True
        caps.has_warp_specialization = True
        caps.recommended_dtype = torch.bfloat16
        caps.recommended_block_size = 512
        caps.max_num_warps = 32
        caps.l2_cache_mb = 192.0

    return caps


def print_gpu_info():
    """Print a nice summary of your GPU's capabilities for Bloth."""
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU found. Bloth requires NVIDIA GPU.")
        return

    caps = detect_gpu()
    mem_used = torch.cuda.memory_allocated() / (1024**3)

    print(f"\n{'='*55}")
    print(f"  🚀 Bloth GPU Configuration")
    print(f"{'='*55}")
    print(f"  GPU:         {caps.name}")
    print(f"  Architecture: {caps.compute_capability}")
    print(f"  VRAM:        {caps.total_memory_gb:.1f} GB total, {mem_used:.2f} GB used")
    print(f"  SMs:         {caps.num_sms}")
    print(f"  L2 Cache:    {caps.l2_cache_mb:.0f} MB")
    print(f"\n  Features Enabled:")
    print(f"    Tensor Cores:      {'✅' if caps.has_tensor_cores else '❌'}")
    print(f"    BF16:              {'✅' if caps.has_bf16 else '❌ (using FP16)'}")
    print(f"    Async Memory Copy: {'✅' if caps.has_async_copy else '❌'}")
    print(f"    FP8 Training:      {'✅' if caps.has_fp8 else '❌'}")
    print(f"    TMA:               {'✅' if caps.has_tma else '❌'}")
    print(f"    Warp Specializ.:   {'✅' if caps.has_warp_specialization else '❌'}")
    print(f"\n  Recommended Settings:")
    print(f"    dtype:      {caps.recommended_dtype}")
    print(f"    block_size: {caps.recommended_block_size}")
    print(f"{'='*55}\n")


# Singleton instance (created once on import)
_gpu_caps: Optional[GPUCapabilities] = None


def get_gpu_caps() -> GPUCapabilities:
    """Get GPU capabilities (cached after first call)."""
    global _gpu_caps
    if _gpu_caps is None:
        _gpu_caps = detect_gpu()
    return _gpu_caps
