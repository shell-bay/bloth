"""
Bloth Training Example
=======================
Shows how to fine-tune any LLM using Bloth.
This script is designed to be run as-is with no manual configuration.

Usage:
    python examples/train_example.py

Requirements:
    pip install bloth transformers peft datasets trl bitsandbytes
"""

import torch
import bloth
from bloth import FastModel

print("""
╔══════════════════════════════════════════════╗
║   BLOTH v2.0 - LLM Fine-Tuning Example      ║
║   Faster than Unsloth. Stable at 128k ctx.  ║
╚══════════════════════════════════════════════╝
""")

# Step 1: Show GPU info
bloth.print_gpu_info()
caps = bloth.get_gpu_caps()

# ── Configuration (auto-adjusted for your GPU) ──────────────────
MODEL_NAME     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # Small model for demo
MAX_SEQ_LEN    = 2048
LORA_RANK      = 16
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LEARNING_RATE  = 2e-4
MAX_STEPS      = 60
OUTPUT_DIR     = "./bloth_output"

# Auto-select dtype based on GPU
DTYPE = caps.recommended_dtype
USE_4BIT = True  # QLoRA by default (saves lots of VRAM)

print(f"📋 Config:")
print(f"   Model:      {MODEL_NAME}")
print(f"   dtype:      {DTYPE}")
print(f"   4-bit QLoRA: {USE_4BIT}")
print(f"   LoRA rank:  {LORA_RANK}")
print(f"   Batch size: {BATCH_SIZE} (x{GRAD_ACCUM} grad accum)")
print()

# Step 2: Load model with Bloth patches applied automatically
try:
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,
        load_in_4bit=USE_4BIT,
    )

    # Step 3: Apply LoRA
    model = FastModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.0,
    )

    # Step 4: Load a sample dataset
    try:
        from datasets import load_dataset
        dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
        print(f"\n📦 Dataset: {len(dataset)} training examples loaded\n")

        # Format the data
        def format_example(example):
            if example.get("input"):
                prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
            else:
                prompt = f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
            return {"text": prompt}

        dataset = dataset.map(format_example)

    except Exception as e:
        print(f"⚠️  Could not load dataset: {e}")
        print("    Creating synthetic dataset for demo...")

        from torch.utils.data import Dataset

        class SyntheticDataset(Dataset):
            def __init__(self, tokenizer, n=100):
                self.data = []
                texts = [
                    "### Instruction:\nExplain what Bloth is.\n\n### Response:\nBloth is a high-performance CUDA kernel library for LLM training.",
                    "### Instruction:\nWhat makes Bloth faster?\n\n### Response:\nBloth fuses multiple GPU operations into single kernels, reducing memory bandwidth usage.",
                ] * (n // 2)

                for text in texts:
                    encoded = tokenizer(
                        text, truncation=True, max_length=512,
                        return_tensors="pt", padding="max_length"
                    )
                    self.data.append({
                        "input_ids": encoded["input_ids"].squeeze(),
                        "attention_mask": encoded["attention_mask"].squeeze(),
                        "labels": encoded["input_ids"].squeeze().clone(),
                    })

            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return self.data[idx]

        dataset = SyntheticDataset(tokenizer)

    # Step 5: Train
    try:
        from trl import SFTTrainer, SFTConfig

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=OUTPUT_DIR,
                max_seq_length=MAX_SEQ_LEN,
                per_device_train_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM,
                warmup_steps=10,
                max_steps=MAX_STEPS,
                learning_rate=LEARNING_RATE,
                fp16=not caps.has_bf16,
                bf16=caps.has_bf16,
                logging_steps=5,
                optim="adamw_8bit",
                save_steps=MAX_STEPS,
                seed=42,
                dataset_text_field="text" if hasattr(dataset, '__iter__') and hasattr(next(iter(dataset)), 'keys') else None,
            ),
        )

        print(f"\n🏃 Starting training for {MAX_STEPS} steps...")
        mem_before = bloth.get_memory_stats()
        print(f"   VRAM before training: {mem_before['allocated_gb']:.2f} GB")

        trainer.train()

        mem_after = bloth.get_memory_stats()
        print(f"\n✅ Training complete!")
        print(f"   Peak VRAM usage: {mem_after['allocated_gb']:.2f} GB")

    except ImportError:
        print("\n⚠️  TRL not installed. Run: pip install trl")
        print("   Training setup was successful, just missing TRL for the trainer.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nMake sure you have: pip install transformers peft datasets trl bitsandbytes")
