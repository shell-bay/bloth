"""
Bloth Benchmark Script
Measures TFLOPS and memory bandwidth to verify kernel quality.
Per Gemini's requirement: >70% bandwidth utilization = non-gimmick kernel.
"""

import torch
import bloth
from bloth.utils.benchmark import benchmark_forward
from bloth.utils.device    import print_device_info

print_device_info()

if not torch.cuda.is_available():
    print("No GPU found. Benchmark requires CUDA.")
    exit(0)

DEVICE = "cuda"
DTYPE  = torch.bfloat16

# ── Benchmark 1: RMSNorm ──────────────────────────────────────────────────
print("\n[1] Benchmarking RMSNorm...")
weight = torch.ones(4096, device=DEVICE, dtype=DTYPE)
norm_fn = lambda x: bloth.rms_norm(x.view(-1, 4096), weight)
benchmark_forward(norm_fn, input_shape=(8, 512, 4096))

# ── Benchmark 2: GEMM ─────────────────────────────────────────────────────
print("\n[2] Benchmarking GEMM...")
B = torch.randn(4096, 4096, device=DEVICE, dtype=DTYPE)
gemm_fn = lambda x: bloth.gemm(x.view(-1, 4096), B)
benchmark_forward(gemm_fn, input_shape=(32, 128, 4096))

# ── Benchmark 3: Memory estimation ────────────────────────────────────────
print("\n[3] Memory estimates for common model sizes:")
for model in ["7b", "13b", "70b"]:
    for prec in ["bf16", "int4"]:
        est = bloth.estimate_memory_usage(model, batch_size=4,
                                           sequence_length=2048, precision=prec)
        print(f"  {model} {prec}: ~{est['total_gb']:.0f} GB total VRAM")

print("\n[Bloth] Benchmark complete!")
