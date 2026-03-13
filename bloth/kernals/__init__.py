"""
Bloth Kernels: High-Performance CUDA Operations
===============================================

This module provides optimized CUDA kernels for LLM training:
- GEMM operations with warp specialization
- FlashAttention implementations
- Layer normalization (RMSNorm, LayerNorm)
- LoRA operations
- Activation functions
- Memory optimization primitives

All kernels are automatically selected based on GPU architecture for optimal performance.
"""

import torch
import warnings
from typing import Optional, Tuple, Union

# Try to import CUDA extensions
try:
    from .. import _C
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False
    warnings.warn("CUDA extensions not available. Using PyTorch fallback.")

# Import Triton kernels (always available)
from .triton_kernels import (
    triton_gemm,
    triton_layer_norm,
    triton_softmax,
    triton_cross_entropy,
)

# Device capability detection
def _get_device_capability():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return (0, 0)

_DEVICE_CAPABILITY = _get_device_capability()

def _supports_warp_specialization():
    return _DEVICE_CAPABILITY[0] >= 9  # Hopper+

def _supports_tma():
    return _DEVICE_CAPABILITY[0] >= 9

def _supports_fp8():
    return _DEVICE_CAPABILITY[0] >= 9

# GEMM operations
def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose_a: bool = False,
    transpose_b: bool = False
) -> torch.Tensor:
    """
    Optimized GEMM with automatic kernel selection.
    
    Uses warp specialization + TMA on Hopper/Blackwell,
    async pipeline on Ampere/Ada.
    
    Args:
        a: Input matrix A [M, K] or [K, M] if transpose_a
        b: Input matrix B [K, N] or [N, K] if transpose_b
        c: Optional output matrix [M, N]
        alpha: Scaling factor for A @ B
        beta: Scaling factor for C
        transpose_a: Whether A is transposed
        transpose_b: Whether B is transposed
    
    Returns:
        Output matrix C = alpha * A @ B + beta * C
    """
    if not _CUDA_AVAILABLE or a.numel() == 0:
        # Fallback to PyTorch
        return torch.nn.functional.linear(a, b.t() if not transpose_b else b)
    
    # Auto-select best kernel
    m = a.shape[-2] if not transpose_a else a.shape[-1]
    k = a.shape[-1] if not transpose_a else a.shape[-2]
    n = b.shape[-1] if not transpose_b else b.shape[-2]
    
    # Use SplitK for small M
    if m <= 64 and _DEVICE_CAPABILITY[0] >= 8:
        return gemm_splitk(a, b, c, alpha, beta, transpose_a, transpose_b)
    
    # Use CUDA kernel
    if c is None:
        c = torch.empty(m, n, dtype=a.dtype, device=a.device)
    
    if _CUDA_AVAILABLE:
        _C.gemm_f16(a, b, c, m, n, k, alpha, beta)
    else:
        # Triton fallback
        c = triton_gemm(a, b, c, alpha, beta, transpose_a, transpose_b)
    
    return c

def gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    FP8 GEMM with automatic scaling.
    
    Requires Hopper (SM90) or newer.
    
    Args:
        a: FP8 input matrix A [M, K]
        b: FP8 input matrix B [K, N]
        a_scale: Per-block scale for A
        b_scale: Per-block scale for B
        c: Optional output matrix [M, N]
        alpha: Additional scaling factor
    
    Returns:
        Output matrix in FP16/BF16
    """
    if not _supports_fp8():
        raise RuntimeError("FP8 requires Hopper (SM90) or newer GPU")
    
    if c is None:
        c = torch.empty(a.shape[0], b.shape[1], dtype=torch.float16, device=a.device)
    
    if _CUDA_AVAILABLE:
        _C.gemm_fp8(a, b, c, a_scale, b_scale, alpha)
    
    return c

def gemm_splitk(
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    transpose_a: bool = False,
    transpose_b: bool = False,
    split_k_factor: Optional[int] = None
) -> torch.Tensor:
    """
    SplitK GEMM for improved occupancy on small batches.
    
    Divides K dimension across multiple thread blocks for better parallelism.
    
    Args:
        split_k_factor: Number of K splits (auto-selected if None)
    """
    if split_k_factor is None:
        # Auto-select based on problem size
        m = a.shape[-2] if not transpose_a else a.shape[-1]
        split_k_factor = 4 if m <= 32 else 2 if m <= 64 else 1
    
    if c is None:
        m = a.shape[-2] if not transpose_a else a.shape[-1]
        n = b.shape[-1] if not transpose_b else b.shape[-2]
        c = torch.empty(m, n, dtype=a.dtype, device=a.device)
    
    if _CUDA_AVAILABLE:
        _C.gemm_splitk(a, b, c, alpha, beta, split_k_factor)
    else:
        c = triton_gemm(a, b, c, alpha, beta, transpose_a, transpose_b)
    
    return c

# Attention operations
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_softmax: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Fused FlashAttention forward pass.
    
    Memory-efficient attention with O(sqrt(N)) memory complexity.
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        v: Value tensor [batch, num_heads, seq_len, head_dim]
        softmax_scale: Scale factor (default: 1/sqrt(head_dim))
        causal: Whether to apply causal mask
        return_softmax: Whether to return softmax output
    
    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
        Optional softmax tensor if return_softmax=True
    """
    batch, num_heads, seq_len, head_dim = q.shape
    
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # Reshape for kernel
    q_reshaped = q.reshape(batch * num_heads, seq_len, head_dim)
    k_reshaped = k.reshape(batch * num_heads, seq_len, head_dim)
    v_reshaped = v.reshape(batch * num_heads, seq_len, head_dim)
    
    output = torch.empty_like(q_reshaped)
    
    if _CUDA_AVAILABLE:
        _C.flash_attention_forward(
            q_reshaped, k_reshaped, v_reshaped, output,
            softmax_scale, batch, num_heads, seq_len, head_dim
        )
    else:
        # PyTorch fallback
        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v).reshape(batch * num_heads, seq_len, head_dim)
    
    output = output.reshape(batch, num_heads, seq_len, head_dim)
    
    if return_softmax:
        # Would need to return softmax from kernel
        return output, None
    return output

def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False
) -> torch.Tensor:
    """
    FlashAttention for variable sequence lengths (packed sequences).
    
    Args:
        cu_seqlens: Cumulative sequence lengths [batch + 1]
        max_seqlen: Maximum sequence length in batch
    
    Returns:
        Output tensor
    """
    batch = cu_seqlens.shape[0] - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)
    
    output = torch.empty_like(q)
    
    if _CUDA_AVAILABLE:
        _C.flash_attention_varlen_forward(
            q, k, v, output, cu_seqlens,
            softmax_scale, batch, num_heads, head_dim, max_seqlen
        )
    
    return output

# Normalization operations
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    RMS Normalization (used in LLaMA, Mistral, Qwen).
    
    Formula: y = x / sqrt(mean(x^2) + eps) * weight
    
    Args:
        x: Input tensor [..., hidden_size]
        weight: Scale weight [hidden_size]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor
    """
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    num_tokens, hidden_size = x.shape
    
    output = torch.empty_like(x)
    inv_rms = torch.empty(num_tokens, dtype=torch.float32, device=x.device)
    
    if _CUDA_AVAILABLE:
        _C.rms_norm_forward(x, weight, output, inv_rms, eps, num_tokens, hidden_size)
    else:
        # PyTorch fallback
        variance = x.pow(2).mean(-1, keepdim=True)
        inv_rms_val = torch.rsqrt(variance + eps)
        output = x * inv_rms_val * weight
    
    return output.reshape(orig_shape)

def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Standard Layer Normalization.
    
    Formula: y = (x - mean) / sqrt(var + eps) * weight + bias
    
    Args:
        x: Input tensor [..., hidden_size]
        weight: Scale weight [hidden_size]
        bias: Shift bias [hidden_size]
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor
    """
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1])
    num_tokens, hidden_size = x.shape
    
    output = torch.empty_like(x)
    mean = torch.empty(num_tokens, dtype=torch.float32, device=x.device)
    inv_std = torch.empty(num_tokens, dtype=torch.float32, device=x.device)
    
    if _CUDA_AVAILABLE:
        _C.layer_norm_forward(x, weight, bias, output, mean, inv_std, eps, num_tokens, hidden_size)
    else:
        # PyTorch fallback
        mean_val = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        inv_std_val = torch.rsqrt(var + eps)
        output = (x - mean_val) * inv_std_val * weight + bias
    
    return output.reshape(orig_shape)

def fused_norm_activation(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    eps: float = 1e-6,
    activation: str = "swiglu"
) -> torch.Tensor:
    """
    Fused RMSNorm + Activation (SwiGLU/GeGLU).
    
    Combines normalization and MLP gate/up projection for efficiency.
    
    Args:
        x: Input tensor [num_tokens, hidden_size]
        norm_weight: RMSNorm weight [hidden_size]
        gate_up_weight: Combined gate/up weight [hidden_size, 2 * intermediate_size]
        eps: Small constant for numerical stability
        activation: "swiglu" or "geglu"
    
    Returns:
        Output tensor [num_tokens, intermediate_size]
    """
    num_tokens, hidden_size = x.shape
    intermediate_size = gate_up_weight.shape[1] // 2
    
    output = torch.empty(num_tokens, intermediate_size, dtype=x.dtype, device=x.device)
    
    if _CUDA_AVAILABLE and activation == "swiglu":
        _C.fused_rms_norm_swiglu(
            x, norm_weight, gate_up_weight, output,
            eps, num_tokens, hidden_size, intermediate_size
        )
    else:
        # PyTorch fallback
        # RMSNorm
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps) * norm_weight
        
        # Gate/Up projection
        gate_up = torch.matmul(x_norm, gate_up_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        
        # SwiGLU
        if activation == "swiglu":
            output = torch.nn.functional.silu(gate) * up
        else:  # geglu
            output = torch.nn.functional.gelu(gate) * up
    
    return output

# LoRA operations
def lora_linear(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float = 1.0
) -> torch.Tensor:
    """
    Fused LoRA linear layer: y = x @ W_base + x @ A @ B * scaling
    
    Args:
        x: Input tensor
        base_weight: Base model weight
        lora_a: LoRA A matrix (low rank)
        lora_b: LoRA B matrix (low rank)
        scaling: LoRA scaling factor
    
    Returns:
        Output tensor
    """
    # Compute base output
    output = torch.nn.functional.linear(x, base_weight)
    
    # Compute LoRA path: x @ A @ B
    lora_output = torch.nn.functional.linear(
        torch.nn.functional.linear(x, lora_a),
        lora_b
    ) * scaling
    
    return output + lora_output

def lora_mlp(
    x: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    lora_gate_a: torch.Tensor,
    lora_gate_b: torch.Tensor,
    lora_up_a: torch.Tensor,
    lora_up_b: torch.Tensor,
    lora_down_a: torch.Tensor,
    lora_down_b: torch.Tensor,
    scaling: float = 1.0
) -> torch.Tensor:
    """
    Fused LoRA MLP (SwiGLU variant).
    
    Args:
        x: Input tensor
        gate_proj_weight: Gate projection weight
        up_proj_weight: Up projection weight
        down_proj_weight: Down projection weight
        lora_*: LoRA matrices for each projection
        scaling: LoRA scaling factor
    
    Returns:
        Output tensor
    """
    # Gate projection with LoRA
    gate = lora_linear(x, gate_proj_weight, lora_gate_a, lora_gate_b, scaling)
    
    # Up projection with LoRA
    up = lora_linear(x, up_proj_weight, lora_up_a, lora_up_b, scaling)
    
    # SwiGLU activation
    activated = torch.nn.functional.silu(gate) * up
    
    # Down projection with LoRA
    output = lora_linear(activated, down_proj_weight, lora_down_a, lora_down_b, scaling)
    
    return output

def fused_lora_attention(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    lora_q_a: torch.Tensor,
    lora_q_b: torch.Tensor,
    lora_k_a: torch.Tensor,
    lora_k_b: torch.Tensor,
    lora_v_a: torch.Tensor,
    lora_v_b: torch.Tensor,
    lora_o_a: torch.Tensor,
    lora_o_b: torch.Tensor,
    num_heads: int,
    head_dim: int,
    scaling: float = 1.0
) -> torch.Tensor:
    """
    Fused LoRA self-attention.
    
    Args:
        hidden_states: Input tensor [batch, seq_len, hidden_size]
        *_proj_weight: Base projection weights
        lora_*_a, lora_*_b: LoRA matrices
        num_heads: Number of attention heads
        head_dim: Dimension per head
        scaling: LoRA scaling factor
    
    Returns:
        Attention output
    """
    batch, seq_len, hidden_size = hidden_states.shape
    
    # Q, K, V projections with LoRA
    q = lora_linear(hidden_states, q_proj_weight, lora_q_a, lora_q_b, scaling)
    k = lora_linear(hidden_states, k_proj_weight, lora_k_a, lora_k_b, scaling)
    v = lora_linear(hidden_states, v_proj_weight, lora_v_a, lora_v_b, scaling)
    
    # Reshape for attention
    q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Flash attention
    attn_output = flash_attention(q, k, v)
    
    # Reshape back
    attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, hidden_size)
    
    # Output projection with LoRA
    output = lora_linear(attn_output, o_proj_weight, lora_o_a, lora_o_b, scaling)
    
    return output

# Embedding operations
def rope_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Rotary Position Embedding (RoPE).
    
    Args:
        x: Input tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine cache [max_seq_len, head_dim//2]
        sin: Sine cache [max_seq_len, head_dim//2]
        position_ids: Optional position indices [batch, seq_len]
    
    Returns:
        Tensor with RoPE applied
    """
    batch, num_heads, seq_len, head_dim = x.shape
    
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
    
    # Gather cos/sin for positions
    cos = cos[position_ids].unsqueeze(2)  # [batch, seq_len, 1, head_dim//2]
    sin = sin[position_ids].unsqueeze(2)
    
    # Split x into pairs
    x1 = x[..., ::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    # Apply rotation
    rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).flatten(-2)
    
    return rotated

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Apply RoPE to Q and K tensors."""
    return rope_embedding(q, cos, sin, position_ids), rope_embedding(k, cos, sin, position_ids)

# Loss functions
def fused_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Fused cross-entropy loss with log-sum-exp trick.
    
    More memory efficient than PyTorch's implementation.
    
    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        reduction: "mean", "sum", or "none"
        label_smoothing: Label smoothing factor
        ignore_index: Index to ignore in loss computation
    
    Returns:
        Loss value
    """
    if _CUDA_AVAILABLE:
        return _C.fused_cross_entropy(logits, labels, reduction, label_smoothing, ignore_index)
    
    # PyTorch fallback
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction=reduction,
        label_smoothing=label_smoothing,
        ignore_index=ignore_index
    )

def fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Fused linear + cross-entropy (for language modeling head).
    
    Avoids materializing full logits tensor, saving memory.
    
    Args:
        hidden_states: [batch, seq_len, hidden_size]
        weight: [vocab_size, hidden_size]
        labels: [batch, seq_len]
    
    Returns:
        Loss value
    """
    if _CUDA_AVAILABLE:
        return _C.fused_linear_cross_entropy(hidden_states, weight, labels, reduction)
    
    # PyTorch fallback
    logits = torch.matmul(hidden_states, weight.t())
    return fused_cross_entropy(logits, labels, reduction)

# Activation functions
def swiglu(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: x * sigmoid(x)"""
    x, gate = x.chunk(2, dim=-1)
    return x * torch.nn.functional.silu(gate)

def geglu(x: torch.Tensor) -> torch.Tensor:
    """GeGLU activation: x * gelu(x)"""
    x, gate = x.chunk(2, dim=-1)
    return x * torch.nn.functional.gelu(gate)

def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation followed by multiplication."""
    return torch.nn.functional.silu(x)

def gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """GELU activation followed by multiplication."""
    return torch.nn.functional.gelu(x)

# Quantization operations
def quantize_weight(
    weight: torch.Tensor,
    bits: int = 4,
    group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight to lower precision.
    
    Args:
        weight: Weight tensor to quantize
        bits: Number of bits (4 or 8)
        group_size: Group size for quantization
    
    Returns:
        Tuple of (quantized_weight, scales)
    """
    if _CUDA_AVAILABLE:
        return _C.quantize_weight(weight, bits, group_size)
    
    # PyTorch fallback (simple per-channel quantization)
    scales = weight.abs().max(dim=-1, keepdim=True)[0] / (2 ** (bits - 1) - 1)
    quantized = (weight / scales).round().clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    return quantized.to(torch.int8), scales

def dequantize_weight(
    quantized_weight: torch.Tensor,
    scales: torch.Tensor,
    bits: int = 4
) -> torch.Tensor:
    """Dequantize weight back to FP16/FP32."""
    if _CUDA_AVAILABLE:
        return _C.dequantize_weight(quantized_weight, scales, bits)
    
    return quantized_weight.float() * scales

def fused_cast_transpose(
    x: torch.Tensor,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused cast + transpose operation.
    
    Efficient for FP8 conversion with transposition.
    
    Args:
        x: Input tensor
        dtype: Target dtype
    
    Returns:
        Tuple of (casted, transposed)
    """
    if _CUDA_AVAILABLE:
        return _C.fused_cast_transpose(x, dtype)
    
    return x.to(dtype), x.t().to(dtype)

# Softmax operations
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Optimized softmax."""
    if _CUDA_AVAILABLE:
        return _C.softmax(x, dim)
    return torch.nn.functional.softmax(x, dim=dim)

def fused_softmax_dropout(
    x: torch.Tensor,
    p: float = 0.0,
    training: bool = True,
    dim: int = -1
) -> torch.Tensor:
    """Fused softmax + dropout for attention."""
    if _CUDA_AVAILABLE and p > 0 and training:
        return _C.fused_softmax_dropout(x, p, dim)
    
    out = torch.nn.functional.softmax(x, dim=dim)
    if p > 0 and training:
        out = torch.nn.functional.dropout(out, p=p)
    return out

__all__ = [
    # GEMM
    'gemm',
    'gemm_fp8',
    'gemm_splitk',
    
    # Attention
    'flash_attention',
    'flash_attention_varlen',
    
    # Normalization
    'rms_norm',
    'layer_norm',
    'fused_norm_activation',
    
    # LoRA
    'lora_linear',
    'lora_mlp',
    'fused_lora_attention',
    
    # Embeddings
    'rope_embedding',
    'apply_rotary_pos_emb',
    
    # Loss
    'fused_cross_entropy',
    'fused_linear_cross_entropy',
    
    # Activations
    'swiglu',
    'geglu',
    'silu_and_mul',
    'gelu_and_mul',
    
    # Quantization
    'quantize_weight',
    'dequantize_weight',
    'fused_cast_transpose',
    
    # Softmax
    'softmax',
    'fused_softmax_dropout',
]
