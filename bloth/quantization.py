"""
Bloth Quantization: Efficient Model Quantization
================================================

Provides quantization support for memory-efficient training:
- 4-bit quantization (QLoRA/Normal Float 4)
- 8-bit quantization (LLM.int8())
- FP8 quantization (Hopper+)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import warnings


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
    device: Optional[str] = None
) -> nn.Module:
    """
    Quantize a model to lower precision.
    
    Args:
        model: Model to quantize
        bits: Number of bits (4 or 8)
        group_size: Group size for quantization
        device: Device to quantize on
    
    Returns:
        Quantized model
    """
    try:
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        
        # Replace linear layers with quantized versions
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get parent and attribute name
                *parent_path, attr_name = name.split('.')
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
                
                # Create quantized linear
                if bits == 4:
                    quantized = Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=torch.float16,
                        compress_statistics=True,
                        quant_type="nf4",
                    )
                elif bits == 8:
                    quantized = Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False,
                    )
                else:
                    raise ValueError(f"Unsupported bits: {bits}")
                
                # Copy weights
                quantized.weight.data = module.weight.data
                if module.bias is not None:
                    quantized.bias.data = module.bias.data
                
                # Replace module
                setattr(parent, attr_name, quantized)
        
        return model
    
    except ImportError:
        warnings.warn("bitsandbytes not available. Quantization skipped.")
        return model


def dequantize_model(model: nn.Module) -> nn.Module:
    """
    Dequantize a quantized model back to FP16/FP32.
    
    Args:
        model: Quantized model
    
    Returns:
        Dequantized model
    """
    try:
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        
        for name, module in model.named_modules():
            if isinstance(module, (Linear4bit, Linear8bitLt)):
                # Get parent
                *parent_path, attr_name = name.split('.')
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)
                
                # Create standard linear
                dequantized = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
                
                # Dequantize and copy weights
                if hasattr(module, 'weight'):
                    weight = module.weight
                    if hasattr(weight, 'CB'):
                        # Dequantize from bitsandbytes format
                        dequantized.weight.data = weight.CB.dequantize()
                    else:
                        dequantized.weight.data = weight.data
                
                if module.bias is not None:
                    dequantized.bias.data = module.bias.data
                
                setattr(parent, attr_name, dequantized)
        
        return model
    
    except ImportError:
        return model


def load_in_4bit(
    model_name: str,
    **kwargs
) -> nn.Module:
    """
    Load a model in 4-bit precision.
    
    Args:
        model_name: Hugging Face model name
        **kwargs: Additional arguments
    
    Returns:
        4-bit quantized model
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        **kwargs
    )
    
    return model


def load_in_8bit(
    model_name: str,
    **kwargs
) -> nn.Module:
    """
    Load a model in 8-bit precision.
    
    Args:
        model_name: Hugging Face model name
        **kwargs: Additional arguments
    
    Returns:
        8-bit quantized model
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        **kwargs
    )
    
    return model


class FP8Quantizer:
    """
    FP8 Quantization for Hopper GPUs.
    
    Uses E4M3 format for weights and E5M2 for gradients.
    """
    
    def __init__(self, amax_history_len: int = 1024):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            raise RuntimeError("FP8 requires Hopper (SM90) or newer GPU")
        
        self.amax_history_len = amax_history_len
        self.amax_history = []
    
    def quantize(
        self,
        x: torch.Tensor,
        dtype: torch.dtype = torch.float8_e4m3fn
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to FP8.
        
        Args:
            x: Input tensor
            dtype: Target FP8 dtype
        
        Returns:
            Tuple of (quantized_tensor, scale)
        """
        # Compute amax (absolute max)
        amax = x.abs().max()
        
        # Update history
        self.amax_history.append(amax.item())
        if len(self.amax_history) > self.amax_history_len:
            self.amax_history.pop(0)
        
        # Compute scale
        max_value = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
        scale = max_value / amax.clamp(min=1e-12)
        
        # Quantize
        quantized = (x * scale).to(dtype)
        
        return quantized, scale
    
    def dequantize(
        self,
        x: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize FP8 tensor.
        
        Args:
            x: Quantized tensor
            scale: Scale factor
        
        Returns:
            Dequantized tensor
        """
        return x.float() / scale


__all__ = [
    'quantize_model',
    'dequantize_model',
    'load_in_4bit',
    'load_in_8bit',
    'FP8Quantizer',
]
