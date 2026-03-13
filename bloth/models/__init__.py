"""
Bloth Models: Universal Model Patching System
=============================================

This module provides automatic patching for ANY transformer model architecture,
making it work with Bloth's optimized kernels.

Unlike Unsloth which requires manual patches for each model,
Bloth uses dynamic analysis to automatically optimize any model.
"""

import torch
import torch.nn as nn
from typing import Type, Dict, Any, Optional, Callable, List, Union
from contextlib import contextmanager
import importlib
import inspect
import warnings

# Registry of patched modules
_PATCH_REGISTRY: Dict[str, Dict[str, Any]] = {}
_CUSTOM_ARCHITECTURES: Dict[str, Dict[str, Any]] = {}

class AutoPatches:
    """
    Automatic model patching using dynamic analysis.
    
    This class analyzes model architecture at runtime and applies
    appropriate optimizations without requiring manual patches.
    """
    
    # Patterns to identify common layer types
    ATTENTION_PATTERNS = [
        'attention', 'attn', 'self_attn', 'cross_attn',
        'MultiHeadAttention', 'Attention'
    ]
    
    MLP_PATTERNS = [
        'mlp', 'feed_forward', 'ffn', 'dense',
        'MLP', 'FeedForward'
    ]
    
    NORM_PATTERNS = [
        'norm', 'ln', 'rms_norm', 'layer_norm',
        'LayerNorm', 'RMSNorm'
    ]
    
    EMBEDDING_PATTERNS = [
        'embed', 'embedding', 'token_embed',
        'Embedding', 'TokenEmbedding'
    ]
    
    @classmethod
    def analyze_model(cls, model: nn.Module) -> Dict[str, Any]:
        """
        Analyze model architecture and identify optimizable components.
        
        Returns:
            Dictionary with identified components and their paths
        """
        analysis = {
            'attention_layers': [],
            'mlp_layers': [],
            'norm_layers': [],
            'embedding_layers': [],
            'linear_layers': [],
            'other_layers': []
        }
        
        for name, module in model.named_modules():
            module_type = type(module).__name__.lower()
            
            # Identify attention layers
            if any(pattern in name.lower() or pattern in module_type for pattern in cls.ATTENTION_PATTERNS):
                analysis['attention_layers'].append((name, module))
            
            # Identify MLP layers
            elif any(pattern in name.lower() or pattern in module_type for pattern in cls.MLP_PATTERNS):
                analysis['mlp_layers'].append((name, module))
            
            # Identify norm layers
            elif any(pattern in name.lower() or pattern in module_type for pattern in cls.NORM_PATTERNS):
                analysis['norm_layers'].append((name, module))
            
            # Identify embedding layers
            elif any(pattern in name.lower() or pattern in module_type for pattern in cls.EMBEDDING_PATTERNS):
                analysis['embedding_layers'].append((name, module))
            
            # Identify linear layers (for potential quantization)
            elif isinstance(module, nn.Linear):
                analysis['linear_layers'].append((name, module))
            
            else:
                analysis['other_layers'].append((name, module))
        
        return analysis
    
    @classmethod
    def patch_attention(cls, module: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """
        Patch attention layer to use FlashAttention.
        
        Args:
            module: Attention module to patch
            config: Configuration dict with head_dim, num_heads, etc.
        
        Returns:
            Patched module
        """
        from ..kernels import flash_attention, apply_rotary_pos_emb
        
        # Store original forward
        original_forward = module.forward
        
        # Extract configuration
        num_heads = config.get('num_heads', getattr(module, 'num_heads', 8))
        head_dim = config.get('head_dim', getattr(module, 'head_dim', 64))
        
        def patched_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs
        ):
            # Get Q, K, V projections
            batch, seq_len, _ = hidden_states.shape
            
            # Check if module has separate q, k, v projections
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                query = module.q_proj(hidden_states)
                key = module.k_proj(hidden_states)
                value = module.v_proj(hidden_states)
            elif hasattr(module, 'qkv_proj'):
                qkv = module.qkv_proj(hidden_states)
                query, key, value = qkv.chunk(3, dim=-1)
            else:
                # Fall back to original forward
                return original_forward(
                    hidden_states, attention_mask, position_ids,
                    past_key_value, output_attentions, use_cache, **kwargs
                )
            
            # Reshape for multi-head attention
            query = query.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            key = key.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            value = value.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Apply RoPE if available
            if hasattr(module, 'rotary_emb') and position_ids is not None:
                cos, sin = module.rotary_emb(value, seq_len=seq_len)
                query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
            
            # Use FlashAttention
            attn_output = flash_attention(query, key, value, causal=True)
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
            
            # Output projection
            if hasattr(module, 'o_proj'):
                attn_output = module.o_proj(attn_output)
            elif hasattr(module, 'out_proj'):
                attn_output = module.out_proj(attn_output)
            
            if output_attentions:
                return (attn_output, None)
            
            return attn_output
        
        module.forward = patched_forward
        module._bloth_patched = True
        
        return module
    
    @classmethod
    def patch_mlp(cls, module: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """
        Patch MLP layer to use fused operations.
        
        Args:
            module: MLP module to patch
            config: Configuration dict
        
        Returns:
            Patched module
        """
        from ..kernels import swiglu, geglu
        
        original_forward = module.forward
        
        # Detect activation type
        activation = config.get('activation', 'swiglu')
        
        def patched_forward(hidden_states):
            # Check for gate_proj/up_proj pattern (LLaMA-style)
            if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj'):
                gate = module.gate_proj(hidden_states)
                up = module.up_proj(hidden_states)
                
                if activation == 'swiglu':
                    hidden_states = swiglu(torch.cat([up, gate], dim=-1))
                else:
                    hidden_states = geglu(torch.cat([up, gate], dim=-1))
                
                hidden_states = module.down_proj(hidden_states)
                return hidden_states
            
            # Check for fc1/fc2 pattern
            elif hasattr(module, 'fc1') and hasattr(module, 'fc2'):
                hidden_states = module.fc1(hidden_states)
                hidden_states = torch.nn.functional.gelu(hidden_states)
                hidden_states = module.fc2(hidden_states)
                return hidden_states
            
            # Fall back to original
            return original_forward(hidden_states)
        
        module.forward = patched_forward
        module._bloth_patched = True
        
        return module
    
    @classmethod
    def patch_norm(cls, module: nn.Module, config: Dict[str, Any]) -> nn.Module:
        """
        Patch normalization layer to use optimized kernels.
        
        Args:
            module: Norm module to patch
            config: Configuration dict
        
        Returns:
            Patched module
        """
        from ..kernels import rms_norm, layer_norm
        
        original_forward = module.forward
        module_type = type(module).__name__
        
        # Check if it's RMSNorm
        is_rms_norm = 'rms' in module_type.lower() or 'RMS' in module_type
        
        def patched_forward(hidden_states):
            if is_rms_norm:
                return rms_norm(hidden_states, module.weight, getattr(module, 'eps', 1e-6))
            else:
                return layer_norm(
                    hidden_states,
                    module.weight,
                    getattr(module, 'bias', None),
                    getattr(module, 'eps', 1e-5)
                )
        
        module.forward = patched_forward
        module._bloth_patched = True
        
        return module
    
    @classmethod
    def apply_patches(cls, model: nn.Module, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Apply all applicable patches to a model.
        
        Args:
            model: Model to patch
            config: Optional configuration overrides
        
        Returns:
            Patched model
        """
        if config is None:
            config = {}
        
        # Analyze model
        analysis = cls.analyze_model(model)
        
        # Patch attention layers
        for name, module in analysis['attention_layers']:
            try:
                cls.patch_attention(module, config.get('attention', {}))
            except Exception as e:
                warnings.warn(f"Failed to patch attention layer {name}: {e}")
        
        # Patch MLP layers
        for name, module in analysis['mlp_layers']:
            try:
                cls.patch_mlp(module, config.get('mlp', {}))
            except Exception as e:
                warnings.warn(f"Failed to patch MLP layer {name}: {e}")
        
        # Patch norm layers
        for name, module in analysis['norm_layers']:
            try:
                cls.patch_norm(module, config.get('norm', {}))
            except Exception as e:
                warnings.warn(f"Failed to patch norm layer {name}: {e}")
        
        # Mark model as patched
        model._bloth_patched = True
        model._bloth_config = config
        
        return model


def patch_model(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    use_flash_attention: bool = True,
    use_fused_mlp: bool = True,
    use_fused_norm: bool = True,
    quantize: Optional[str] = None,
) -> nn.Module:
    """
    Patch any model to use Bloth optimizations.
    
    This is the main entry point for model patching. It automatically
    detects model architecture and applies appropriate optimizations.
    
    Args:
        model: PyTorch model to patch
        config: Optional configuration dictionary
        use_flash_attention: Whether to use FlashAttention
        use_fused_mlp: Whether to use fused MLP operations
        use_fused_norm: Whether to use fused normalization
        quantize: Optional quantization ('4bit', '8bit', or None)
    
    Returns:
        Patched model (modified in-place)
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> import bloth
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        >>> model = bloth.patch_model(model)
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available. Model patching will have no effect.")
        return model
    
    # Build config
    patch_config = {
        'attention': {
            'use_flash_attention': use_flash_attention,
        },
        'mlp': {
            'use_fused': use_fused_mlp,
        },
        'norm': {
            'use_fused': use_fused_norm,
        }
    }
    
    if config:
        patch_config.update(config)
    
    # Apply quantization if requested
    if quantize:
        from ..quantization import quantize_model
        model = quantize_model(model, bits=4 if quantize == '4bit' else 8)
    
    # Apply patches
    model = AutoPatches.apply_patches(model, patch_config)
    
    # Compile model if available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            warnings.warn(f"torch.compile failed: {e}")
    
    return model


def patch_peft_model(
    model: nn.Module,
    peft_config: Any,
    config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Patch a PEFT (LoRA/QLoRA) model with Bloth optimizations.
    
    Args:
        model: Base model or PEFT model
        peft_config: PEFT configuration (LoraConfig, etc.)
        config: Optional Bloth configuration
    
    Returns:
        Patched PEFT model
    """
    from peft import get_peft_model, LoraConfig
    
    # First patch base model
    model = patch_model(model, config)
    
    # Apply PEFT if not already applied
    if not hasattr(model, 'peft_config'):
        model = get_peft_model(model, peft_config)
    
    return model


def get_model_architecture(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed architecture information about a model.
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with architecture details
    """
    analysis = AutoPatches.analyze_model(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'num_attention_layers': len(analysis['attention_layers']),
        'num_mlp_layers': len(analysis['mlp_layers']),
        'num_norm_layers': len(analysis['norm_layers']),
        'num_linear_layers': len(analysis['linear_layers']),
        'attention_layers': [name for name, _ in analysis['attention_layers']],
        'mlp_layers': [name for name, _ in analysis['mlp_layers']],
        'norm_layers': [name for name, _ in analysis['norm_layers']],
    }


def register_custom_architecture(
    name: str,
    attention_class: Type[nn.Module],
    mlp_class: Type[nn.Module],
    norm_class: Type[nn.Module],
    patch_functions: Optional[Dict[str, Callable]] = None
) -> None:
    """
    Register a custom model architecture for patching.
    
    This allows Bloth to optimize custom or new model architectures
    that aren't explicitly supported.
    
    Args:
        name: Name of the architecture
        attention_class: Attention layer class
        mlp_class: MLP layer class
        norm_class: Normalization layer class
        patch_functions: Optional custom patch functions
    
    Example:
        >>> register_custom_architecture(
        ...     'my_custom_model',
        ...     MyAttention,
        ...     MyMLP,
        ...     MyRMSNorm,
        ... )
    """
    _CUSTOM_ARCHITECTURES[name] = {
        'attention_class': attention_class,
        'mlp_class': mlp_class,
        'norm_class': norm_class,
        'patch_functions': patch_functions or {}
    }
    
    # Add patterns for auto-detection
    AutoPatches.ATTENTION_PATTERNS.append(attention_class.__name__)
    AutoPatches.MLP_PATTERNS.append(mlp_class.__name__)
    AutoPatches.NORM_PATTERNS.append(norm_class.__name__)


def is_patched(model: nn.Module) -> bool:
    """Check if a model has been patched by Bloth."""
    return getattr(model, '_bloth_patched', False)


def unpatch_model(model: nn.Module) -> nn.Module:
    """
    Remove Bloth patches from a model.
    
    Note: This may not perfectly restore the original model if
    quantization was applied.
    
    Args:
        model: Patched model
    
    Returns:
        Unpatched model
    """
    # This is a best-effort unpatching
    # In practice, it's often easier to reload the original model
    
    for name, module in model.named_modules():
        if getattr(module, '_bloth_patched', False):
            # Remove patch marker
            delattr(module, '_bloth_patched')
            
            # Note: The forward function cannot be easily restored
            # without keeping a reference to the original
    
    if hasattr(model, '_bloth_patched'):
        delattr(model, '_bloth_patched')
    
    return model


__all__ = [
    'AutoPatches',
    'patch_model',
    'patch_peft_model',
    'get_model_architecture',
    'register_custom_architecture',
    'is_patched',
    'unpatch_model',
]
