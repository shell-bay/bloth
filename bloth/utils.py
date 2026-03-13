"""
Bloth Utilities: Helper Functions
=================================

Various utility functions for memory management, optimization, and debugging.
"""

import torch
import torch.nn as nn
import random
import numpy as np
from typing import Optional, Dict, Any, Tuple
import warnings


def get_gpu_memory() -> Dict[str, float]:
    """
    Get GPU memory statistics.
    
    Returns:
        Dictionary with memory stats in MB
    """
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated': torch.cuda.memory_allocated() / 1024**2,
        'reserved': torch.cuda.memory_reserved() / 1024**2,
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,
        'max_reserved': torch.cuda.max_memory_reserved() / 1024**2,
        'free': (torch.cuda.get_device_properties(0).total_memory - 
                 torch.cuda.memory_allocated()) / 1024**2,
    }


def print_gpu_memory(prefix: str = "") -> None:
    """Print GPU memory statistics."""
    mem = get_gpu_memory()
    if prefix:
        print(f"{prefix} ", end="")
    print(f"GPU Memory: Allocated={mem.get('allocated', 0):.1f}MB, "
          f"Reserved={mem.get('reserved', 0):.1f}MB, "
          f"Free={mem.get('free', 0):.1f}MB")


def optimize_memory() -> None:
    """
    Optimize GPU memory usage.
    
    This function:
    - Clears CUDA cache
    - Runs garbage collection
    - Sets optimal memory settings
    """
    import gc
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Run garbage collection
    gc.collect()
    
    # Set optimal memory settings
    if torch.cuda.is_available():
        # Enable memory efficient attention
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Enable gradient checkpointing for a model.
    
    Args:
        model: Model to enable checkpointing for
    
    Returns:
        Model with checkpointing enabled
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    else:
        # Try to enable on underlying model
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
    
    return model


def disable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """Disable gradient checkpointing."""
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    else:
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = False
    
    return model


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimal_batch_size(
    model: nn.Module,
    sequence_length: int,
    hidden_size: int,
    vocab_size: int,
    gpu_memory_gb: Optional[float] = None
) -> int:
    """
    Estimate optimal batch size based on available GPU memory.
    
    Args:
        model: Model to estimate for
        sequence_length: Sequence length
        hidden_size: Hidden dimension
        vocab_size: Vocabulary size
        gpu_memory_gb: Available GPU memory in GB (auto-detected if None)
    
    Returns:
        Estimated optimal batch size
    """
    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif gpu_memory_gb is None:
        return 1
    
    # Rough estimation of memory per sample
    # This is a simplified calculation
    bytes_per_param = 2  # FP16
    
    # Model parameters
    model_params = sum(p.numel() for p in model.parameters())
    model_memory = model_params * bytes_per_param / (1024**3)
    
    # Activations memory (rough estimate)
    activation_memory = (
        sequence_length * hidden_size * 12 * bytes_per_param / (1024**3)
    )
    
    # Available memory for batch
    available_memory = gpu_memory_gb * 0.8  # Use 80% of memory
    available_memory -= model_memory
    
    if available_memory <= 0:
        return 1
    
    optimal_batch = int(available_memory / activation_memory)
    
    return max(1, min(optimal_batch, 128))


def estimate_memory_usage(
    model_name: str,
    batch_size: int,
    sequence_length: int,
    precision: str = "fp16"
) -> Dict[str, float]:
    """
    Estimate memory usage for training.
    
    Args:
        model_name: Model name or size (e.g., "7b", "13b")
        batch_size: Batch size
        sequence_length: Sequence length
        precision: Precision ("fp32", "fp16", "bf16", "fp8", "int8", "int4")
    
    Returns:
        Dictionary with memory estimates in GB
    """
    # Parse model size
    if isinstance(model_name, str):
        if 'b' in model_name.lower():
            size_str = model_name.lower().replace('b', '')
            try:
                model_size_gb = float(size_str) * 2  # Roughly 2GB per billion params in FP16
            except:
                model_size_gb = 7 * 2
        else:
            model_size_gb = 14  # Default to 7B model
    else:
        model_size_gb = 14
    
    # Precision multiplier
    precision_mult = {
        'fp32': 4.0,
        'fp16': 2.0,
        'bf16': 2.0,
        'fp8': 1.0,
        'int8': 1.0,
        'int4': 0.5,
    }.get(precision, 2.0)
    
    # Model memory
    model_memory = model_size_gb * precision_mult
    
    # Activation memory (rough estimate)
    activation_memory = (
        batch_size * sequence_length * 4096 * 12 * precision_mult / (1024**3)
    )
    
    # Optimizer memory (Adam uses 2x model size for states)
    optimizer_memory = model_memory * 2
    
    # Gradients
    gradient_memory = model_memory
    
    # Total
    total_memory = model_memory + activation_memory + optimizer_memory + gradient_memory
    
    return {
        'model': model_memory,
        'activations': activation_memory,
        'optimizer': optimizer_memory,
        'gradients': gradient_memory,
        'total': total_memory,
    }


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layers
    num_attention = sum(1 for _ in model.named_modules() if 'attention' in type(_[1]).__name__.lower())
    num_linear = sum(1 for _ in model.named_modules() if isinstance(_[1], nn.Linear))
    num_norm = sum(1 for _ in model.named_modules() if 'norm' in type(_[1]).__name__.lower())
    
    # Model size in GB
    model_size_gb = total_params * 2 / (1024**3)  # FP16
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_gb': model_size_gb,
        'num_attention_layers': num_attention,
        'num_linear_layers': num_linear,
        'num_norm_layers': num_norm,
    }


def print_model_info(model: nn.Module) -> None:
    """Print model information."""
    info = get_model_info(model)
    
    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total Parameters:     {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"Non-trainable:        {info['non_trainable_parameters']:,}")
    print(f"Model Size (FP16):    {info['model_size_gb']:.2f} GB")
    print(f"Attention Layers:     {info['num_attention_layers']}")
    print(f"Linear Layers:        {info['num_linear_layers']}")
    print(f"Norm Layers:          {info['num_norm_layers']}")
    print("=" * 60)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.
    
    Args:
        model: Model to count
        trainable_only: Only count trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_parameters(model: nn.Module, pattern: Optional[str] = None) -> nn.Module:
    """
    Freeze model parameters.
    
    Args:
        model: Model to freeze
        pattern: Optional pattern to match parameter names
    
    Returns:
        Model with frozen parameters
    """
    for name, param in model.named_parameters():
        if pattern is None or pattern in name:
            param.requires_grad = False
    
    return model


def unfreeze_parameters(model: nn.Module, pattern: Optional[str] = None) -> nn.Module:
    """Unfreeze model parameters."""
    for name, param in model.named_parameters():
        if pattern is None or pattern in name:
            param.requires_grad = True
    
    return model


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpus': [],
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['gpus'].append({
                'id': i,
                'name': props.name,
                'total_memory_gb': props.total_memory / (1024**3),
                'compute_capability': f"{props.major}.{props.minor}",
            })
    
    return info


def print_device_info() -> None:
    """Print device information."""
    info = get_device_info()
    
    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"Number of GPUs: {info['num_gpus']}")
    
    for gpu in info['gpus']:
        print(f"\nGPU {gpu['id']}: {gpu['name']}")
        print(f"  Total Memory: {gpu['total_memory_gb']:.2f} GB")
        print(f"  Compute Capability: {gpu['compute_capability']}")
    
    print("=" * 60)


def benchmark_forward(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Benchmark model forward pass.
    
    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_iterations: Number of iterations
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with benchmark results
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape, device=device)
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    end.record()
    
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    avg_time_ms = elapsed_ms / num_iterations
    
    return {
        'total_time_ms': elapsed_ms,
        'avg_time_ms': avg_time_ms,
        'throughput_samples_per_sec': input_shape[0] * num_iterations / (elapsed_ms / 1000),
    }


def clear_cache() -> None:
    """Clear all caches."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


__all__ = [
    'get_gpu_memory',
    'print_gpu_memory',
    'optimize_memory',
    'enable_gradient_checkpointing',
    'disable_gradient_checkpointing',
    'set_seed',
    'get_optimal_batch_size',
    'estimate_memory_usage',
    'get_model_info',
    'print_model_info',
    'count_parameters',
    'freeze_parameters',
    'unfreeze_parameters',
    'get_device_info',
    'print_device_info',
    'benchmark_forward',
    'clear_cache',
]
