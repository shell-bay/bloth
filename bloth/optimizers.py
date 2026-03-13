"""
Bloth Optimizers: Memory-Efficient Optimizers
=============================================

Provides memory-efficient optimizers for LLM training:
- 8-bit AdamW
- FP8 AdamW (Hopper+)
- 8-bit Lion
"""

import torch
from torch.optim import Optimizer
from typing import Any, Dict, List, Optional, Tuple
import warnings


class AdamW8bit(Optimizer):
    """
    8-bit AdamW optimizer.
    
    Reduces optimizer state memory by 2x compared to standard AdamW
    by quantizing momentum states to 8-bit.
    
    Args:
        params: Model parameters
        lr: Learning rate
        betas: Coefficients for running averages
        eps: Term added for numerical stability
        weight_decay: Weight decay coefficient
        quantize_block_size: Block size for quantization
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        quantize_block_size: int = 2048,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if betas[0] < 0.0 or betas[0] > 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if betas[1] < 0.0 or betas[1] > 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            quantize_block_size=quantize_block_size,
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW8bit does not support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Quantized momentum buffers
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.uint8)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.uint8)
                    state['exp_avg_scale'] = torch.ones(1, device=p.device)
                    state['exp_avg_sq_scale'] = torch.ones(1, device=p.device)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_scale = state['exp_avg_scale']
                exp_avg_sq_scale = state['exp_avg_sq_scale']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                # Dequantize -> update -> quantize
                exp_avg_fp32 = exp_avg.float() * exp_avg_scale
                exp_avg_sq_fp32 = exp_avg_sq.float() * exp_avg_sq_scale
                
                exp_avg_fp32.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq_fp32.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Quantize back
                exp_avg_scale = exp_avg_fp32.abs().max() / 127.0
                exp_avg_sq_scale = exp_avg_sq_fp32.abs().max() / 127.0
                
                exp_avg.copy_((exp_avg_fp32 / exp_avg_scale).clamp(-128, 127).to(torch.uint8))
                exp_avg_sq.copy_((exp_avg_sq_fp32 / exp_avg_sq_scale).clamp(-128, 127).to(torch.uint8))
                
                state['exp_avg_scale'] = exp_avg_scale
                state['exp_avg_sq_scale'] = exp_avg_sq_scale
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                
                # Compute denominator
                denom = (exp_avg_sq_fp32.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                
                # Update parameters
                p.data.addcdiv_(exp_avg_fp32, denom, value=-step_size)
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss


class AdamWFp8(Optimizer):
    """
    FP8 AdamW optimizer for Hopper GPUs.
    
    Uses FP8 E4M3 for momentum states, reducing memory by 4x.
    Requires Hopper (SM90) or newer GPU.
    
    Args:
        params: Model parameters
        lr: Learning rate
        betas: Coefficients for running averages
        eps: Term added for numerical stability
        weight_decay: Weight decay coefficient
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            raise RuntimeError("AdamWFp8 requires Hopper (SM90) or newer GPU")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.float8_e4m3fn)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.float8_e4m3fn)
                    state['exp_avg_scale'] = torch.ones(1, device=p.device)
                    state['exp_avg_sq_scale'] = torch.ones(1, device=p.device)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Dequantize, update, quantize
                exp_avg_fp32 = exp_avg.float() * state['exp_avg_scale']
                exp_avg_sq_fp32 = exp_avg_sq.float() * state['exp_avg_sq_scale']
                
                exp_avg_fp32.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq_fp32.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Quantize to FP8
                state['exp_avg_scale'] = exp_avg_fp32.abs().max() / 448.0
                state['exp_avg_sq_scale'] = exp_avg_sq_fp32.abs().max() / 448.0
                
                exp_avg.copy_((exp_avg_fp32 / state['exp_avg_scale']).to(torch.float8_e4m3fn))
                exp_avg_sq.copy_((exp_avg_sq_fp32 / state['exp_avg_sq_scale']).to(torch.float8_e4m3fn))
                
                # Bias correction and update
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq_fp32.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                
                p.data.addcdiv_(exp_avg_fp32, denom, value=-step_size)
                
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss


class Lion8bit(Optimizer):
    """
    8-bit Lion optimizer.
    
    Lion (EvoLved Sign Momentum) is a memory-efficient optimizer
    that only tracks momentum, not second moments.
    
    Reference: Chen et al. "Symbolic Discovery of Optimization Algorithms" (2023)
    
    Args:
        params: Model parameters
        lr: Learning rate
        betas: Coefficients for running averages
        weight_decay: Weight decay coefficient
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0.0 or betas[0] > 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if betas[1] < 0.0 or betas[1] > 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data, dtype=torch.uint8)
                    state['momentum_scale'] = torch.ones(1, device=p.device)
                
                momentum = state['momentum']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Dequantize momentum
                momentum_fp32 = momentum.float() * state['momentum_scale']
                
                # Update momentum: m = beta1 * m + (1 - beta1) * grad
                momentum_fp32.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Quantize back
                state['momentum_scale'] = momentum_fp32.abs().max() / 127.0
                momentum.copy_((momentum_fp32 / state['momentum_scale']).clamp(-128, 127).to(torch.uint8))
                
                # Compute update: c = beta1 * m + (1 - beta1) * grad (interpolated momentum)
                update = beta1 * momentum_fp32 + (1 - beta1) * grad
                
                # Lion uses sign of update
                update = update.sign()
                
                # Update parameters
                p.data.add_(update, alpha=-group['lr'])
                
                # Weight decay (decoupled)
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss


__all__ = [
    'AdamW8bit',
    'AdamWFp8',
    'Lion8bit',
]
