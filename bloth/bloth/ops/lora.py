"""
Bloth LoRA op — optimized Low-Rank Adaptation kernels.
Fuses the base linear + LoRA A + LoRA B into minimal memory reads.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def _lora_linear_fwd_kernel(
    Out_ptr, X_ptr, Base_ptr, LoraA_ptr, LoraB_ptr,
    M, N, K, R,
    scaling,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes: out = x @ base.T + scaling * (x @ lora_a.T) @ lora_b.T
    In a single tiled loop to reduce HBM traffic.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_base = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Base weight GEMM
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        mask_k = k_offs < K

        x_tile = tl.load(
            X_ptr + offs_m[:, None] * K + k_offs[None, :],
            mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0,
        )
        w_tile = tl.load(
            Base_ptr + offs_n[:, None] * K + k_offs[None, :],
            mask=(offs_n[:, None] < N) & mask_k[None, :], other=0.0,
        )
        acc_base += tl.dot(x_tile, tl.trans(w_tile))

    # LoRA path: x @ A^T → intermediate [M, R], then @ B^T → [M, N]
    # We compute this inline to avoid storing the intermediate rank tensor
    acc_lora = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for r in range(0, R, 16):
        r_offs = r + tl.arange(0, 16)
        mask_r = r_offs < R

        xa_tile = tl.zeros([BLOCK_M, 16], dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_offs = k + offs_k
            x_t  = tl.load(X_ptr    + offs_m[:, None] * K + (k + offs_k)[None, :],
                            mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
            la_t = tl.load(LoraA_ptr + r_offs[:, None] * K + (k + offs_k)[None, :],
                            mask=mask_r[:, None] & ((k + offs_k)[None, :] < K), other=0.0)
            xa_tile += tl.dot(x_t, tl.trans(la_t))

        lb_tile = tl.load(
            LoraB_ptr + offs_n[:, None] * R + r_offs[None, :],
            mask=(offs_n[:, None] < N) & mask_r[None, :], other=0.0,
        )
        acc_lora += tl.dot(xa_tile, tl.trans(lb_tile))

    out = acc_base + scaling * acc_lora
    tl.store(
        Out_ptr + offs_m[:, None] * N + offs_n[None, :],
        out.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def lora_linear(x, base_w, lora_a, lora_b, scaling=1.0):
    """
    Fused base linear + LoRA: out = x @ W^T + scaling * x @ A^T @ B^T
    """
    M, K = x.shape
    N    = base_w.shape[0]
    R    = lora_a.shape[0]
    out  = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)

    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    _lora_linear_fwd_kernel[grid](
        out, x.contiguous(), base_w.contiguous(),
        lora_a.contiguous(), lora_b.contiguous(),
        M, N, K, R, scaling,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_warps=4,
    )
    return out


def lora_mlp(x, gate_w, up_w, down_w, lora_weights, scaling=1.0):
    """
    LoRA MLP with SwiGLU: out = (gate(x) * silu * up(x)) @ down^T
    lora_weights: dict with keys 'gate_a','gate_b','up_a','up_b','down_a','down_b'
    """
    gate = lora_linear(x, gate_w, lora_weights["gate_a"], lora_weights["gate_b"], scaling)
    up   = lora_linear(x, up_w,   lora_weights["up_a"],   lora_weights["up_b"],   scaling)
    hidden = torch.nn.functional.silu(gate) * up
    return lora_linear(hidden, down_w, lora_weights["down_a"], lora_weights["down_b"], scaling)


class BlothLoRA(nn.Module):
    def __init__(self, in_features, out_features, r=16, lora_alpha=32, dropout=0.05):
        super().__init__()
        self.r      = r
        self.scaling = lora_alpha / r
        self.lora_A  = nn.Parameter(torch.randn(r, in_features) * (1 / math.sqrt(r)))
        self.lora_B  = nn.Parameter(torch.zeros(out_features, r))
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x, base_weight):
        x_drop = self.dropout(x)
        return lora_linear(x_drop, base_weight, self.lora_A, self.lora_B, self.scaling)
