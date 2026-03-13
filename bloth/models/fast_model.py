"""
Bloth FastModel: High-Level API for Fast Training
=================================================

Provides a simple, Unsloth-like API for loading and training models
with automatic Bloth optimizations.

Example:
    >>> from bloth import FastModel
    >>> model, tokenizer = FastModel.from_pretrained("meta-llama/Llama-2-7b")
    >>> model = model.get_peft_model(r=16, lora_alpha=32)
    >>> trainer = bloth.Trainer(model=model, ...)
    >>> trainer.train()
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from pathlib import Path
import warnings
import gc

from . import patch_model, get_model_architecture, is_patched


class FastModel:
    """
    High-level API for fast model loading and training.
    
    This class wraps a transformer model with Bloth optimizations
    and provides convenient methods for PEFT, quantization, and training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FastModel.
        
        Args:
            model: The underlying transformer model
            tokenizer: Tokenizer for the model
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        self._peft_model = None
        self._is_quantized = False
        
        # Store original model for reference
        self._base_model = model
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        max_seq_length: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        full_finetuning: bool = False,
        device_map: Union[str, Dict[str, Any]] = "auto",
        trust_remote_code: bool = False,
        use_flash_attention: bool = True,
        use_fused_mlp: bool = True,
        use_fused_norm: bool = True,
        rope_scaling: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Tuple["FastModel", Any]:
        """
        Load a pretrained model with Bloth optimizations.
        
        This is the main entry point for using Bloth. It loads a model
        from Hugging Face Hub and automatically applies all optimizations.
        
        Args:
            model_name: Hugging Face model name or path
            max_seq_length: Maximum sequence length (for RoPE scaling)
            dtype: Data type (auto-detected if None)
            load_in_4bit: Load model in 4-bit precision (QLoRA)
            load_in_8bit: Load model in 8-bit precision
            full_finetuning: Enable full finetuning (no quantization)
            device_map: Device mapping for model loading
            trust_remote_code: Trust remote code in model
            use_flash_attention: Use FlashAttention
            use_fused_mlp: Use fused MLP operations
            use_fused_norm: Use fused normalization
            rope_scaling: RoPE scaling configuration
            **kwargs: Additional arguments for model loading
        
        Returns:
            Tuple of (FastModel, tokenizer)
        
        Example:
            >>> from bloth import FastModel
            >>> model, tokenizer = FastModel.from_pretrained(
            ...     "meta-llama/Llama-2-7b-hf",
            ...     max_seq_length=4096,
            ...     load_in_4bit=True,
            ... )
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Setup quantization config
        quantization_config = None
        if load_in_4bit and not full_finetuning:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit and not full_finetuning:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=dtype,
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            rope_scaling=rope_scaling,
            **kwargs
        )
        
        # Apply Bloth patches
        model = patch_model(
            model,
            use_flash_attention=use_flash_attention,
            use_fused_mlp=use_fused_mlp,
            use_fused_norm=use_fused_norm,
        )
        
        # Create FastModel instance
        config = {
            'max_seq_length': max_seq_length,
            'dtype': dtype,
            'load_in_4bit': load_in_4bit,
            'load_in_8bit': load_in_8bit,
            'full_finetuning': full_finetuning,
        }
        
        fast_model = cls(model, tokenizer, config)
        fast_model._is_quantized = load_in_4bit or load_in_8bit
        
        return fast_model, tokenizer
    
    def get_peft_model(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: bool = True,
        random_state: int = 3407,
        max_seq_length: Optional[int] = None,
        use_rslora: bool = False,
        init_lora_weights: Union[bool, str] = True,
        **kwargs
    ) -> "FastModel":
        """
        Apply PEFT (LoRA) to the model.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha (scaling)
            target_modules: Modules to apply LoRA to (auto-detected if None)
            lora_dropout: LoRA dropout rate
            bias: Bias training mode ("none", "all", "lora_only")
            use_gradient_checkpointing: Enable gradient checkpointing
            random_state: Random seed
            max_seq_length: Maximum sequence length
            use_rslora: Use Rank-Stabilized LoRA
            init_lora_weights: Initialize LoRA weights
            **kwargs: Additional PEFT arguments
        
        Returns:
            Self with PEFT applied
        """
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Auto-detect target modules if not specified
        if target_modules is None:
            target_modules = self._detect_target_modules()
        
        # Calculate LoRA scaling
        if use_rslora:
            # Rank-Stabilized LoRA: scale by sqrt(r)
            lora_scale = lora_alpha / (r ** 0.5)
        else:
            lora_scale = lora_alpha / r
        
        # Create LoRA config
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
            init_lora_weights=init_lora_weights,
            **kwargs
        )
        
        # Apply PEFT
        self._peft_model = get_peft_model(self.model, peft_config)
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self._peft_model.gradient_checkpointing_enable()
        
        # Update model reference
        self.model = self._peft_model
        
        # Update config
        self.config.update({
            'lora_r': r,
            'lora_alpha': lora_alpha,
            'lora_scale': lora_scale,
            'target_modules': target_modules,
            'use_rslora': use_rslora,
        })
        
        return self
    
    def _detect_target_modules(self) -> List[str]:
        """Auto-detect target modules for LoRA based on model architecture."""
        model_type = self.model.config.model_type if hasattr(self.model, 'config') else 'unknown'
        
        # Common patterns for different model types
        module_patterns = {
            'llama': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'mistral': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'qwen': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'qwen2': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'gemma': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            'phi': ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            'gpt2': ["c_attn", "c_proj", "c_fc"],
            'gptj': ["q_proj", "v_proj", "fc_in", "fc_out"],
            'falcon': ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            'mpt': ["Wqkv", "out_proj", "up_proj", "down_proj"],
        }
        
        # Get modules for this model type
        target_modules = module_patterns.get(model_type, ["q_proj", "v_proj"])
        
        # Verify modules exist in model
        available_modules = []
        for name, _ in self.model.named_modules():
            for target in target_modules:
                if target in name:
                    available_modules.append(target)
                    break
        
        # Return unique modules
        return list(set(available_modules)) if available_modules else ["q_proj", "v_proj"]
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text with optimized inference.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to sample
            **kwargs: Additional generation arguments
        
        Returns:
            Generated token IDs
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                use_cache=True,
                **kwargs
            )
        
        return outputs
    
    def save_model(
        self,
        save_path: Union[str, Path],
        save_tokenizer: bool = True,
        save_adapter: bool = True,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        hub_token: Optional[str] = None,
    ) -> None:
        """
        Save the model (and optionally tokenizer) to disk.
        
        Args:
            save_path: Path to save to
            save_tokenizer: Whether to save tokenizer
            save_adapter: Whether to save only LoRA adapter (not full model)
            push_to_hub: Whether to push to Hugging Face Hub
            hub_model_id: Hub model ID (if different from save_path)
            hub_token: Hugging Face token
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if save_adapter and self._peft_model is not None:
            # Save only LoRA adapter
            self._peft_model.save_pretrained(save_path)
        else:
            # Save full model
            self.model.save_pretrained(
                save_path,
                push_to_hub=push_to_hub,
                repo_id=hub_model_id,
                token=hub_token,
            )
        
        if save_tokenizer:
            self.tokenizer.save_pretrained(save_path)
    
    def merge_and_unload(self) -> nn.Module:
        """
        Merge LoRA weights into base model and unload PEFT.
        
        Returns:
            Merged model
        """
        if self._peft_model is None:
            warnings.warn("No PEFT model to merge")
            return self.model
        
        merged_model = self._peft_model.merge_and_unload()
        self.model = merged_model
        self._peft_model = None
        
        return merged_model
    
    def print_trainable_parameters(self) -> None:
        """Print information about trainable parameters."""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_params:,} || "
              f"Trainable%: {100 * trainable_params / all_params:.4f}")
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        if not torch.cuda.is_available():
            return {'gpu_memory': 0, 'allocated': 0, 'reserved': 0}
        
        return {
            'gpu_memory': torch.cuda.memory_allocated() / 1024**2,
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'reserved': torch.cuda.memory_reserved() / 1024**2,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,
        }
    
    def __getattr__(self, name: str):
        """Forward attribute access to underlying model."""
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def __call__(self, *args, **kwargs):
        """Forward call to underlying model."""
        return self.model(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"FastModel({self.model.__class__.__name__}, patched={is_patched(self.model)})"


# Convenience function for quick model loading
def load_model(
    model_name: str,
    **kwargs
) -> Tuple[FastModel, Any]:
    """
    Quick load a model with Bloth optimizations.
    
    This is a convenience wrapper around FastModel.from_pretrained().
    
    Args:
        model_name: Hugging Face model name
        **kwargs: Additional arguments for from_pretrained
    
    Returns:
        Tuple of (FastModel, tokenizer)
    
    Example:
        >>> from bloth import load_model
        >>> model, tokenizer = load_model("meta-llama/Llama-2-7b", load_in_4bit=True)
    """
    return FastModel.from_pretrained(model_name, **kwargs)


__all__ = [
    'FastModel',
    'load_model',
]
