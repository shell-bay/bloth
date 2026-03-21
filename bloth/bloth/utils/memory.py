"""
Bloth Memory Utilities
Estimate, monitor, and optimise GPU VRAM usage.
"""
import torch
import math


def get_gpu_memory(device_id: int = 0) -> dict:
    """Return current VRAM usage in GB."""
    if not torch.cuda.is_available():
        return {"total_gb": 0, "allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}
    total     = torch.cuda.get_device_properties(device_id).total_memory
    allocated = torch.cuda.memory_allocated(device_id)
    reserved  = torch.cuda.memory_reserved(device_id)
    return {
        "total_gb":     round(total     / 1e9, 2),
        "allocated_gb": round(allocated / 1e9, 2),
        "reserved_gb":  round(reserved  / 1e9, 2),
        "free_gb":      round((total - reserved) / 1e9, 2),
    }


def print_gpu_memory(device_id: int = 0):
    m = get_gpu_memory(device_id)
    print(f"  VRAM: {m['allocated_gb']:.1f} GB used / "
          f"{m['total_gb']:.1f} GB total  "
          f"({m['free_gb']:.1f} GB free)")


def optimize_memory():
    """Release unused cached allocations back to CUDA."""
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
    Estimate VRAM needed for training.

    Args:
        model_name      : e.g. "7b", "13b", "70b", or any HF model id
        batch_size      : training batch size per GPU
        sequence_length : context length
        precision       : "fp32", "bf16", "fp16", "fp8", "int8", "int4"
        lora_rank       : LoRA rank (0 = full fine-tune)

    Returns:
        dict with model_gb, activations_gb, optimizer_gb, total_gb
    """
    _size_map = {
        "1b": 1e9, "3b": 3e9, "7b": 7e9, "8b": 8e9,
        "13b": 13e9, "14b": 14e9, "30b": 30e9,
        "34b": 34e9, "70b": 70e9, "72b": 72e9,
    }
    params = next(
        (v for k, v in _size_map.items() if k in model_name.lower()),
        7e9,
    )

    bpp = {"fp32": 4, "fp16": 2, "bf16": 2,
           "fp8":  1, "int8": 1, "int4": 0.5}.get(precision, 2)

    if lora_rank > 0:
        # Frozen weights + small adapters
        model_gb = params * 0.5 / 1e9 + params * 0.01 * (lora_rank / 16) * 4 / 1e9
        opt_gb   = model_gb * 0.05   # only adapter params in optimiser
    else:
        model_gb = params * bpp / 1e9
        opt_gb   = params * 8 / 1e9  # AdamW: 2 fp32 moments

    hidden   = int(math.sqrt(params / 12))
    act_gb   = batch_size * sequence_length * hidden * 4 / 1e9

    total = model_gb + act_gb + opt_gb
    return {
        "model_gb":       round(model_gb, 2),
        "activations_gb": round(act_gb,   2),
        "optimizer_gb":   round(opt_gb,   2),
        "total_gb":       round(total,    2),
    }


def get_optimal_batch_size(
    model,
    sequence_length: int,
    hidden_size: int,
    safety_factor: float = 0.8,
) -> int:
    """Estimate the largest batch that fits in free VRAM."""
    mem   = get_gpu_memory()
    free  = mem["free_gb"] * 1e9 * safety_factor
    bps   = sequence_length * hidden_size * 4 * 2   # fwd + bwd bytes
    raw   = max(1, int(free / bps))
    return 2 ** int(math.log2(raw))
