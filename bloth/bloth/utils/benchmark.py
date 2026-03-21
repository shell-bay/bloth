"""
Bloth Benchmark Utility
Measures latency, throughput, and memory bandwidth utilisation.

Key metric: >70% bandwidth utilisation = kernel is NOT a gimmick.
(Gemini's requirement — enforced here automatically.)
"""
import torch


def benchmark_forward(
    model_or_fn,
    input_shape: tuple,
    num_iterations: int = 100,
    warmup: int = 10,
    dtype=torch.bfloat16,
    device: str = "cuda",
) -> dict:
    """
    Benchmark any model or callable.

    Args:
        model_or_fn    : nn.Module or plain function
        input_shape    : shape tuple, e.g. (4, 512, 4096)
        num_iterations : timed runs
        warmup         : un-timed warmup runs
        dtype          : input tensor dtype
        device         : "cuda" or "cpu"

    Returns:
        dict with timing, bandwidth, and utilisation metrics
    """
    if not torch.cuda.is_available():
        print("[Bloth Benchmark] No CUDA GPU — skipped.")
        return {}

    x = torch.randn(input_shape, dtype=dtype, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            model_or_fn(x)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    times    = []

    for _ in range(num_iterations):
        start_ev.record()
        with torch.no_grad():
            model_or_fn(x)
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    avg_ms = sum(times) / len(times)
    min_ms = min(times)

    n_elems        = 1
    for s in input_shape:
        n_elems *= s
    throughput     = n_elems / (avg_ms / 1000)
    bytes_moved    = n_elems * 2 * 2          # bfloat16 read + write
    bandwidth_gbs  = bytes_moved / (avg_ms / 1000) / 1e9

    props  = torch.cuda.get_device_properties(0)
    sm     = props.major * 10 + props.minor
    _peak  = {80: 2000, 86: 900, 89: 1008, 90: 3350, 100: 8000}
    peak   = _peak.get(sm, 900)
    bw_pct = min(bandwidth_gbs / peak * 100, 100.0)

    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  Bloth Benchmark")
    print(sep)
    print(f"  GPU             : {props.name}")
    print(f"  Input shape     : {input_shape}")
    print(f"  Avg latency     : {avg_ms:.3f} ms")
    print(f"  Min latency     : {min_ms:.3f} ms")
    print(f"  Bandwidth       : {bandwidth_gbs:.1f} GB/s")
    verdict = "✅ Non-gimmick kernel" if bw_pct > 70 else "⚠️  Below 70% — needs tuning"
    print(f"  BW utilisation  : {bw_pct:.1f}%  {verdict}")
    print(f"{sep}\n")

    return {
        "avg_time_ms":               round(avg_ms,       3),
        "min_time_ms":               round(min_ms,       3),
        "throughput_elements_per_s": int(throughput),
        "bandwidth_gb_s":            round(bandwidth_gbs, 1),
        "bandwidth_utilization_pct": round(bw_pct,       1),
        "gpu":                       props.name,
    }
