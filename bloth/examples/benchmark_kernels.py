"""
Bloth v1.0 — Kernel Benchmark
Measures TFLOPS and bandwidth utilisation.
>70% utilisation = non-gimmick kernel (Gemini's requirement).
"""
import torch
import bloth
from bloth.utils.benchmark import benchmark_forward
from bloth.utils.device    import print_device_info

print_device_info()

if not torch.cuda.is_available():
    print("Benchmark requires a CUDA GPU.")
    exit(0)

D = "cuda"

print("\n[1] RMSNorm  (hidden=4096)")
w = torch.ones(4096, device=D, dtype=torch.bfloat16)
benchmark_forward(lambda x: bloth.rms_norm(x.view(-1, 4096), w),
                  input_shape=(8, 512, 4096))

print("\n[2] GEMM  (4096x4096)")
B = torch.randn(4096, 4096, device=D, dtype=torch.bfloat16)
benchmark_forward(lambda x: bloth.gemm(x.view(-1, 4096), B),
                  input_shape=(32, 128, 4096))

print("\n[3] Memory estimates")
for m in ["7b", "13b", "70b"]:
    for p in ["bf16", "int4"]:
        e = bloth.estimate_memory_usage(m, 4, 2048, p)
        print(f"   {m:4s} {p:5s}: ~{e['total_gb']:5.1f} GB total")
