"""
Bloth Memory Utilities
Estimate, optimize, and monitor GPU memory usage.
"""
import torch
import math

# Bytes per element for each dtype
DTYPE_BYTES = {
    torch.float32:     4,
    torch.float16:     2,
    torch.bfloat16:    2,
    torch.int8:        1,
    torch.int4:        0.5,  # approximate
}

def get_gpu_memory(device_id: int = 0) -> dict:
    if not torch.cuda.is_available():
        return {"total": 0, "used": 0, "free": 0}
    total = torch.cuda.get_device_properties(device_id).total_memory
    reserved   = torch.cuda.memory_reserved(device_id)
    allocated  = torch.cuda.memory_allocated(device_id)
    return {
        "total_gb":     round(total     / 1e9, 2),
        "reserved_gb":  round(reserved  / 1e9, 2),
        "allocated_gb": round(allocated / 1e9, 2),
        "free_gb":      round((total - reserved) / 1e9, 2),
    }

def print_gpu_memory(device_id: int = 0):
    m = get_gpu_memory(device_id)
    print(f"  VRAM: {m['allocated_gb']:.1f} GB used / {m['total_gb']:.1f} GB total "
          f"({m['free_gb']:.1f} GB free)")

def optimize_memory():
    """Release unused cached memory back to the OS."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def estimate_memory_usage(
    model_name: str,
    batch_size: int,
    sequence_length: int,
    precision: str = "bf16",
    lora_rank: int = 0,
) -> dict:
    """
    Estimate VRAM required for training a model.
    model_name can be a size string like "7b", "13b", "70b" or a HF model id.
    """
    # Parameter count lookup
    size_map = {
        "1b": 1e9, "3b": 3e9, "7b": 7e9, "8b": 8e9,
        "13b": 13e9, "14b": 14e9, "30b": 30e9,
        "34b": 34e9, "70b": 70e9, "72b": 72e9,
    }
    params = None
    for key, val in size_map.items():
        if key in model_name.lower():
            params = val
            break
    if params is None:
        params = 7e9  # default guess

    bytes_per = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
    bpp = bytes_per.get(precision, 2)

    model_gb   = params * bpp / 1e9
    # With LoRA, only adapters are in full precision
    if lora_rank > 0:
        lora_params = params * 0.01 * lora_rank / 16  # rough estimate
        model_gb   = params * 0.5 / 1e9 + lora_params * 4 / 1e9

    # Activations: hidden_size ≈ sqrt(params/12), 4 bytes per element
    hidden = int(math.sqrt(params / 12))
    act_gb = (batch_size * sequence_length * hidden * 4) / 1e9

    # Optimizer states: AdamW stores 2 moments (fp32) = 8x model params
    opt_gb = 0 if lora_rank > 0 else (params * 8 / 1e9)

    total = model_gb + act_gb + opt_gb
    return {
        "model_gb":      round(model_gb, 2),
        "activations_gb": round(act_gb,  2),
        "optimizer_gb":  round(opt_gb,   2),
        "total_gb":      round(total,    2),
    }

def get_optimal_batch_size(
    model,
    sequence_length: int,
    hidden_size: int,
    safety_factor: float = 0.8,
) -> int:
    """
    Estimate the largest batch size that fits in available VRAM.
    Uses a simple heuristic: 4 bytes * seq * hidden * batch for activations.
    """
    mem = get_gpu_memory()
    free_bytes = mem["free_gb"] * 1e9 * safety_factor
    bytes_per_sample = sequence_length * hidden_size * 4 * 2  # fwd + bwd
    optimal = max(1, int(free_bytes / bytes_per_sample))
    # Round down to power of 2
    return 2 ** int(math.log2(optimal))
