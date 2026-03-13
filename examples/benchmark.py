"""
Bloth Benchmark: Performance Comparison
=======================================

Benchmark Bloth against standard PyTorch and Unsloth.
"""

import torch
import time
import json
from typing import Dict, List
import sys

import bloth
from bloth import FastModel


def benchmark_attention(batch_size: int, seq_len: int, num_heads: int, head_dim: int, num_iters: int = 100) -> Dict[str, float]:
    """Benchmark FlashAttention vs standard attention."""
    print(f"\nBenchmarking Attention (batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = bloth.kernels.flash_attention(q, k, v)
    
    torch.cuda.synchronize()
    
    # Benchmark Bloth FlashAttention
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        _ = bloth.kernels.flash_attention(q, k, v)
    end.record()
    
    torch.cuda.synchronize()
    bloth_time = start.elapsed_time(end) / num_iters
    
    # Benchmark standard attention
    start.record()
    for _ in range(num_iters):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        _ = torch.matmul(attn, v)
    end.record()
    
    torch.cuda.synchronize()
    standard_time = start.elapsed_time(end) / num_iters
    
    speedup = standard_time / bloth_time
    
    print(f"  Standard Attention: {standard_time:.3f} ms")
    print(f"  Bloth FlashAttention: {bloth_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    return {
        'standard_ms': standard_time,
        'bloth_ms': bloth_time,
        'speedup': speedup,
    }


def benchmark_rms_norm(num_tokens: int, hidden_size: int, num_iters: int = 1000) -> Dict[str, float]:
    """Benchmark RMS normalization."""
    print(f"\nBenchmarking RMS Norm (tokens={num_tokens}, hidden={hidden_size})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.float16)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = bloth.kernels.rms_norm(x, weight)
    
    torch.cuda.synchronize()
    
    # Benchmark Bloth
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        _ = bloth.kernels.rms_norm(x, weight)
    end.record()
    
    torch.cuda.synchronize()
    bloth_time = start.elapsed_time(end) / num_iters
    
    # Benchmark PyTorch
    start.record()
    for _ in range(num_iters):
        variance = x.pow(2).mean(-1, keepdim=True)
        _ = x * torch.rsqrt(variance + 1e-6) * weight
    end.record()
    
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / num_iters
    
    speedup = pytorch_time / bloth_time
    
    print(f"  PyTorch RMS Norm: {pytorch_time:.3f} ms")
    print(f"  Bloth RMS Norm: {bloth_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    return {
        'pytorch_ms': pytorch_time,
        'bloth_ms': bloth_time,
        'speedup': speedup,
    }


def benchmark_gemm(m: int, n: int, k: int, num_iters: int = 100) -> Dict[str, float]:
    """Benchmark GEMM."""
    print(f"\nBenchmarking GEMM (M={m}, N={n}, K={k})")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        _ = bloth.kernels.gemm(a, b)
    
    torch.cuda.synchronize()
    
    # Benchmark Bloth
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        _ = bloth.kernels.gemm(a, b)
    end.record()
    
    torch.cuda.synchronize()
    bloth_time = start.elapsed_time(end) / num_iters
    
    # Benchmark PyTorch (cuBLAS)
    start.record()
    for _ in range(num_iters):
        _ = torch.matmul(a, b)
    end.record()
    
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / num_iters
    
    speedup = pytorch_time / bloth_time
    
    print(f"  PyTorch/cuBLAS GEMM: {pytorch_time:.3f} ms")
    print(f"  Bloth GEMM: {bloth_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    return {
        'pytorch_ms': pytorch_time,
        'bloth_ms': bloth_time,
        'speedup': speedup,
    }


def main():
    print("=" * 60)
    print("Bloth Performance Benchmark")
    print("=" * 60)
    
    # Print system info
    bloth.print_device_info()
    
    if not torch.cuda.is_available():
        print("\nCUDA not available. Skipping benchmarks.")
        return
    
    results = {}
    
    # Benchmark Attention
    results['attention'] = benchmark_attention(
        batch_size=4,
        seq_len=2048,
        num_heads=32,
        head_dim=128,
        num_iters=50
    )
    
    # Benchmark RMS Norm
    results['rms_norm'] = benchmark_rms_norm(
        num_tokens=4096,
        hidden_size=4096,
        num_iters=500
    )
    
    # Benchmark GEMM
    results['gemm'] = benchmark_gemm(
        m=4096,
        n=4096,
        k=4096,
        num_iters=100
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    for benchmark_name, benchmark_results in results.items():
        print(f"\n{benchmark_name.upper()}:")
        for metric, value in benchmark_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to benchmark_results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
