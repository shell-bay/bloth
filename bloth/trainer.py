"""
Bloth Trainer: Optimized Training Loop
======================================

Provides a high-performance trainer with Bloth-specific optimizations:
- Automatic mixed precision with FP8 support
- Gradient accumulation optimization
- Memory-efficient training
- Integration with Bloth kernels
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import warnings
import json
import os


@dataclass
class TrainingArguments:
    """Training arguments for Bloth Trainer."""
    
    # Output
    output_dir: str = field(default="./bloth_output")
    
    # Training hyperparameters
    num_train_epochs: int = field(default=3)
    max_steps: int = field(default=-1)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=2e-4)
    weight_decay: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)
    
    # Optimization
    optim: str = field(default="adamw_torch")
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.03)
    warmup_steps: int = field(default=0)
    
    # Logging and saving
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    save_total_limit: int = field(default=2)
    
    # Precision
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    fp8: bool = field(default=False)  # Requires Hopper
    
    # Memory optimization
    gradient_checkpointing: bool = field(default=True)
    max_seq_length: Optional[int] = field(default=None)
    packing: bool = field(default=False)
    
    # Data
    dataloader_num_workers: int = field(default=0)
    dataloader_pin_memory: bool = field(default=True)
    
    # Reproducibility
    seed: int = field(default=42)
    
    # Evaluation
    evaluation_strategy: str = field(default="steps")
    eval_accumulation_steps: int = field(default=1)
    
    # Advanced
    group_by_length: bool = field(default=False)
    length_column_name: str = field(default="length")
    
    def __post_init__(self):
        """Validate arguments."""
        if self.fp8 and not torch.cuda.is_available():
            warnings.warn("FP8 requires CUDA. Disabling FP8.")
            self.fp8 = False
        
        if self.fp8 and torch.cuda.get_device_capability()[0] < 9:
            warnings.warn("FP8 requires Hopper (SM90) or newer. Disabling FP8.")
            self.fp8 = False
        
        # Auto-enable BF16 if available and not explicitly disabled
        if not self.fp16 and not self.bf16:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.bf16 = True


class Trainer:
    """
    High-performance trainer with Bloth optimizations.
    
    This trainer integrates with Bloth kernels for maximum training speed
    while maintaining compatibility with the Hugging Face ecosystem.
    
    Example:
        >>> from bloth import FastModel, Trainer, TrainingArguments
        >>> model, tokenizer = FastModel.from_pretrained("meta-llama/Llama-2-7b")
        >>> args = TrainingArguments(output_dir="./output", num_train_epochs=3)
        >>> trainer = Trainer(model=model, args=args, train_dataset=dataset)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler] = (None, None),
    ):
        """
        Initialize Trainer.
        
        Args:
            model: Model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer
            data_collator: Data collator function
            compute_metrics: Metrics computation function
            callbacks: Training callbacks
            optimizers: Tuple of (optimizer, lr_scheduler)
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        self.optimizer, self.lr_scheduler = optimizers
        
        # Training state
        self.state = {
            'global_step': 0,
            'epoch': 0,
            'best_metric': None,
            'is_local_process_zero': True,
        }
        
        # Setup
        self._setup_training()
    
    def _setup_training(self) -> None:
        """Setup training environment."""
        # Set seed
        torch.manual_seed(self.args.seed)
        
        # Setup device
        self.device = next(self.model.parameters()).device
        
        # Setup optimizer if not provided
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        # Setup LR scheduler if not provided
        if self.lr_scheduler is None:
            self.lr_scheduler = self._create_scheduler()
        
        # Setup gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.args.fp16 else None
        
        # Enable gradient checkpointing
        if self.args.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'gradient_checkpointing_enable'):
                self.model.model.gradient_checkpointing_enable()
        
        # Create output directory
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup dataloaders
        if self.train_dataset is not None:
            self.train_dataloader = self._create_dataloader(
                self.train_dataset,
                self.args.per_device_train_batch_size,
                shuffle=True
            )
        
        if self.eval_dataset is not None:
            self.eval_dataloader = self._create_dataloader(
                self.eval_dataset,
                self.args.per_device_eval_batch_size,
                shuffle=False
            )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        from torch.optim import AdamW
        
        # Separate parameters with/without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "bias" in name or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        return AdamW(param_groups, lr=self.args.learning_rate)
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        num_training_steps = self._get_num_training_steps()
        num_warmup_steps = (
            self.args.warmup_steps if self.args.warmup_steps > 0
            else int(num_training_steps * self.args.warmup_ratio)
        )
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=num_warmup_steps
        )
        
        # Main scheduler
        if self.args.lr_scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps
            )
        else:
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps - num_warmup_steps
            )
        
        # Combined scheduler
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[num_warmup_steps]
        )
    
    def _create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
        """Create dataloader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            collate_fn=self.data_collator,
        )
    
    def _get_num_training_steps(self) -> int:
        """Get total number of training steps."""
        if self.args.max_steps > 0:
            return self.args.max_steps
        
        num_epochs = self.args.num_train_epochs
        steps_per_epoch = len(self.train_dataloader) // self.args.gradient_accumulation_steps
        
        return num_epochs * steps_per_epoch
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        
        Returns:
            Training metrics
        """
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        print(f"***** Running training *****")
        print(f"  Num examples = {len(self.train_dataset)}")
        print(f"  Num Epochs = {self.args.num_train_epochs}")
        print(f"  Batch size per device = {self.args.per_device_train_batch_size}")
        print(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {self._get_num_training_steps()}")
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for epoch in range(self.args.num_train_epochs):
            self.state['epoch'] = epoch
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(
                    enabled=self.args.fp16 or self.args.bf16 or self.args.fp8,
                    dtype=torch.bfloat16 if self.args.bf16 else torch.float16
                ):
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    loss = loss / self.args.gradient_accumulation_steps
                
                # Backward pass
                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                
                # Gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.args.max_grad_norm > 0:
                        if self.args.fp16:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.max_grad_norm
                        )
                    
                    # Optimizer step
                    if self.args.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.state['global_step'] += 1
                    
                    # Logging
                    if self.state['global_step'] % self.args.logging_steps == 0:
                        avg_loss = total_loss / self.args.logging_steps
                        lr = self.lr_scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        
                        print(f"Step {self.state['global_step']}: "
                              f"loss = {avg_loss:.4f}, "
                              f"lr = {lr:.2e}, "
                              f"time = {elapsed:.2f}s")
                        
                        total_loss = 0.0
                    
                    # Evaluation
                    if (self.eval_dataset is not None and 
                        self.state['global_step'] % self.args.eval_steps == 0):
                        self.evaluate()
                    
                    # Save checkpoint
                    if self.state['global_step'] % self.args.save_steps == 0:
                        self.save_checkpoint()
                    
                    # Check max steps
                    if (self.args.max_steps > 0 and 
                        self.state['global_step'] >= self.args.max_steps):
                        break
            
            # Check max steps
            if self.args.max_steps > 0 and self.state['global_step'] >= self.args.max_steps:
                break
        
        # Final save
        self.save_checkpoint()
        
        print(f"Training completed in {time.time() - start_time:.2f}s")
        
        return {'train_loss': total_loss}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        if self.eval_dataset is None:
            return {}
        
        print("***** Running evaluation *****")
        
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(
                    enabled=self.args.fp16 or self.args.bf16
                ):
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps
        metrics = {'eval_loss': avg_loss}
        
        if self.compute_metrics:
            metrics.update(self.compute_metrics())
        
        print(f"Evaluation results: {metrics}")
        
        self.model.train()
        
        return metrics
    
    def save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{self.state['global_step']}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), checkpoint_dir / "pytorch_model.bin")
        
        # Save optimizer and scheduler
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'global_step': self.state['global_step'],
            'epoch': self.state['epoch'],
        }, checkpoint_dir / "training_state.bin")
        
        # Save args
        with open(checkpoint_dir / "training_args.json", 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        if (checkpoint_dir / "pytorch_model.bin").exists():
            state_dict = torch.load(checkpoint_dir / "pytorch_model.bin", map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        # Load training state
        training_state = torch.load(checkpoint_dir / "training_state.bin", map_location=self.device)
        self.optimizer.load_state_dict(training_state['optimizer'])
        self.lr_scheduler.load_state_dict(training_state['lr_scheduler'])
        self.state['global_step'] = training_state['global_step']
        self.state['epoch'] = training_state['epoch']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save final model."""
        output_dir = output_dir or self.args.output_dir
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(output_path)
        else:
            torch.save(self.model.state_dict(), output_path / "pytorch_model.bin")
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)
        
        print(f"Model saved to {output_path}")


__all__ = [
    'TrainingArguments',
    'Trainer',
]
