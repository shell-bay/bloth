"""
Bloth Kernel 3: Standalone RMSNorm
===================================
Used as building block AND exported as public API.
float32 accumulator ensures gradient stability even at fp8/bf16.
"""

import triton
import triton.language as tl
import torch


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16),
    ],
    key=["n_cols"],
)
@triton.jit
def _rms_norm_fwd_kernel(
    Y_ptr, X_ptr, W_ptr, RMS_ptr,
    stride_x, stride_y,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
    SAVE_RMS: tl.constexpr,   # whether to save rms for backward
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs,                  mask=mask, other=1.0).to(tl.float32)

    # float32 variance accumulation (critical for numerical stability)
    variance = tl.sum(x * x, axis=0) / n_cols
    rms_inv  = 1.0 / tl.sqrt(variance + eps)

    y = x * rms_inv * w

    if SAVE_RMS:
        tl.store(RMS_ptr + row, rms_inv)

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _rms_norm_bwd_kernel(
    DX_ptr, DW_ptr,
    DY_ptr, X_ptr, W_ptr, RMS_ptr,
    stride_x, stride_dy,
    n_cols, M,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    dy      = tl.load(DY_ptr  + row * stride_dy + offs, mask=mask, other=0.0).to(tl.float32)
    x       = tl.load(X_ptr   + row * stride_x  + offs, mask=mask, other=0.0).to(tl.float32)
    w       = tl.load(W_ptr   + offs,                   mask=mask, other=1.0).to(tl.float32)
    rms_inv = tl.load(RMS_ptr + row).to(tl.float32)

    x_norm = x * rms_inv

    # dL/dw = sum over batch of dy * x_norm
    tl.atomic_add(DW_ptr + offs, (dy * x_norm).to(tl.float32), mask=mask)

    # dL/dx = rms_inv * (dy*w - x_norm * dot(dy*w, x_norm)/n_cols)
    dy_w = dy * w
    dot  = tl.sum(dy_w * x_norm, axis=0) / n_cols
    dx   = rms_inv * (dy_w - x_norm * dot)
    tl.store(DX_ptr + row * stride_x + offs, dx.to(tl.bfloat16), mask=mask)


class _RMSNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        M, N = x.shape
        y    = torch.empty_like(x)
        rms  = torch.empty(M, device=x.device, dtype=torch.float32)
        grid = (M,)

        _rms_norm_fwd_kernel[grid](
            y, x, weight, rms,
            x.stride(0), y.stride(0),
            N, eps, SAVE_RMS=True,
        )
        ctx.save_for_backward(x, weight, rms)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rms = ctx.saved_tensors
        M, N = x.shape
        dx = torch.empty_like(x)
        dw = torch.zeros(N, dtype=torch.float32, device=x.device)
        BLOCK = min(triton.next_power_of_2(N), 4096)

        _rms_norm_bwd_kernel[(M,)](
            dx, dw, dy.contiguous(), x, weight, rms,
            x.stride(0), dy.stride(0),
            N, M,
            BLOCK_SIZE=BLOCK,
            num_warps=8 if N >= 2048 else 4,
        )
        return dx, dw.to(weight.dtype), None


def rms_norm(x, weight, eps=1e-6):
    """
    Bloth RMSNorm — drop-in replacement for transformers.models.llama.modeling_llama.LlamaRMSNorm

    Args:
        x      : input tensor [M, N]  (any dtype, computed in float32)
        weight : learnable scale [N]
        eps    : small constant for numerical stability

    Returns:
        normalized tensor, same shape and dtype as x
    """
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])

    result = _RMSNormFunction.apply(x.contiguous(), weight.contiguous(), eps)
    return result.view(orig_shape)
