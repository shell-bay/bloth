"""
Bloth Kernel: Adaptive Delayed FP8 Scaling
============================================
FP8 is 2x faster than BF16 but causes NaN loss spikes without proper scaling.

Solution: Rolling Amax history buffer (default: 16 steps).
  - Stores max absolute value from last N training steps
  - Uses history max to compute scale instead of per-step max
  - Smooths out sudden spikes that would cause NaN loss
  - Result: FP8 speed + BF16 stability
"""

import torch
import triton
import triton.language as tl


class AdaptiveDelayedScaler:
    """
    Manages FP8 scaling with circular Amax history buffer.
    One instance per tensor/weight matrix.
    """
    FP8_MAX = 448.0  # max representable in float8_e4m3fn

    def __init__(self, history_len: int = 16, device: str = "cuda"):
        self.history_len  = history_len
        self.amax_history = torch.zeros(history_len,
                                         device=device if torch.cuda.is_available() else "cpu")
        self.ptr          = 0
        self.scale        = torch.ones(1,
                                        device=device if torch.cuda.is_available() else "cpu")
        self.scale_inv    = torch.ones(1,
                                        device=device if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def update(self, tensor: torch.Tensor):
        """Call after forward pass. Updates Amax history and recomputes scale."""
        amax = tensor.detach().abs().float().max()
        self.amax_history[self.ptr % self.history_len] = amax
        self.ptr += 1

        max_amax = self.amax_history.max()
        new_scale = self.FP8_MAX / (max_amax.clamp(min=1e-12))
        new_scale = new_scale.clamp(max=1e4)

        self.scale     = new_scale
        self.scale_inv = 1.0 / new_scale.clamp(min=1e-12)

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Scale tensor into FP8 range."""
        scaled = tensor.float() * self.scale
        if hasattr(torch, "float8_e4m3fn"):
            return scaled.to(torch.float8_e4m3fn)
        return scaled.clamp(-448.0, 448.0).to(torch.float16)

    def dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() * self.scale_inv


_fp8_scalers: dict = {}


def get_scaler(name: str, device: str = "cuda") -> AdaptiveDelayedScaler:
    """Get or create a named scaler (one per weight matrix)."""
    if name not in _fp8_scalers:
        _fp8_scalers[name] = AdaptiveDelayedScaler(device=device)
    return _fp8_scalers[name]


@triton.jit
def _fp8_quantize_kernel(
    Out_ptr, In_ptr, Scale_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x     = tl.load(In_ptr    + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(Scale_ptr).to(tl.float32)

    x_scaled = tl.clamp(x * scale, -448.0, 448.0)
    tl.store(Out_ptr + offs, x_scaled.to(tl.float16), mask=mask)


def quantize_tensor_fp8(tensor: torch.Tensor, scaler: AdaptiveDelayedScaler) -> torch.Tensor:
    """Fast Triton-accelerated FP8 quantization."""
    flat  = tensor.contiguous().view(-1)
    out   = torch.empty_like(flat, dtype=torch.float16)
    BLOCK = 1024
    grid  = (triton.cdiv(flat.numel(), BLOCK),)

    _fp8_quantize_kernel[grid](
        out, flat, scaler.scale,
        flat.numel(),
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out.view(tensor.shape)
