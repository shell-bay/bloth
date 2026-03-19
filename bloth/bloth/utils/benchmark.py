"""
Bloth Benchmark Utilities
Measures TFLOPS, memory bandwidth utilization, and latency.
Per Gemini's suggestion: a non-gimmick kernel must hit >70% memory bandwidth.
"""
import torch
import time
from typing import Callable, Optional


def _count_flops_attention(batch, heads, seq_len, head_dim) -> int:
    """FLOPs for attention: 2 * batch * heads * seq^2 * head_dim"""
    return 2 * batch * heads * seq_len * seq_len * head_dim


def benchmark_forward(
    model_or_fn,
    input_shape: tuple,
    num_iterations: int = 100,
    warmup: int = 10,
    dtype=torch.bfloat16,
    device: str = "cuda",
) -> dict:
    """
    Benchmark a model or function.

    Args:
        model_or_fn : nn.Module or callable
        input_shape : shape of dummy input tensor
        num_iterations : number of timed runs
        warmup : number of warmup runs (not timed)

    Returns:
        dict with avg_time_ms, throughput, tflops, bandwidth_utilization
    """
    if not torch.cuda.is_available():
        print("[Bloth] No CUDA GPU — benchmark skipped.")
        return {}

    x = torch.randn(input_shape, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model_or_fn(x)
    torch.cuda.synchronize()

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(num_iterations):
        start.record()
        with torch.no_grad():
            out = model_or_fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds

    avg_ms = sum(times) / len(times)
    min_ms = min(times)

    # Elements per second
    total_elements = 1
    for s in input_shape:
        total_elements *= s
    throughput = total_elements / (avg_ms / 1000)

    # Memory bandwidth (bytes read + written / time)
    bytes_per_elem = 2  # bfloat16
    bytes_moved    = total_elements * bytes_per_elem * 2   # read + write
    bandwidth_gbs  = bytes_moved / (avg_ms / 1000) / 1e9

    # Peak memory bandwidth for the GPU
    props = torch.cuda.get_device_properties(0)
    # Rough estimate: A100=2TB/s, H100=3.3TB/s, 4090=1TB/s, 3090=0.9TB/s
    sm = props.major * 10 + props.minor
    peak_bw_gbs = {80: 2000, 89: 1008, 90: 3350, 100: 8000}.get(sm, 900)

    bw_utilization = min(bandwidth_gbs / peak_bw_gbs * 100, 100.0)

    results = {
        "avg_time_ms":               round(avg_ms,        3),
        "min_time_ms":               round(min_ms,        3),
        "throughput_elements_per_s": int(throughput),
        "bandwidth_gb_s":            round(bandwidth_gbs, 1),
        "bandwidth_utilization_pct": round(bw_utilization, 1),
        "gpu":                       props.name,
    }

    print(f"\n{'='*55}")
    print(f"  Bloth Benchmark Results")
    print(f"{'='*55}")
    print(f"  GPU:                   {props.name}")
    print(f"  Input shape:           {input_shape}")
    print(f"  Avg latency:           {avg_ms:.3f} ms")
    print(f"  Min latency:           {min_ms:.3f} ms")
    print(f"  Bandwidth:             {bandwidth_gbs:.1f} GB/s")
    print(f"  Bandwidth utilization: {bw_utilization:.1f}%  {'✅ Good' if bw_utilization > 70 else '⚠️ Needs tuning'}")
    print(f"{'='*55}\n")

    return results
