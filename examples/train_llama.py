"""
Example: Train LLaMA with Bloth Optimizations
=============================================

This example demonstrates how to use Bloth to train a LLaMA model
with 3-5x speedup compared to standard training.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import bloth
from bloth import FastModel, Trainer, TrainingArguments


def main():
    print("=" * 60)
    print("Bloth Training Example: LLaMA-2 7B")
    print("=" * 60)
    
    # Print device info
    bloth.print_device_info()
    
    # Configuration
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    MAX_SEQ_LENGTH = 2048
    
    # Load model with Bloth optimizations
    print("\nLoading model with Bloth optimizations...")
    model, tokenizer = FastModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,  # QLoRA for memory efficiency
        use_flash_attention=True,
        use_fused_mlp=True,
        use_fused_norm=True,
    )
    
    # Print model info
    bloth.print_model_info(model.model)
    
    # Apply LoRA
    print("\nApplying LoRA...")
    model = model.get_peft_model(
        r=16,
        lora_alpha=32,
        target_modules=None,  # Auto-detect
        lora_dropout=0.05,
        use_gradient_checkpointing=True,
        random_state=42,
    )
    model.print_trainable_parameters()
    
    # Load dataset (example using Alpaca)
    print("\nLoading dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    
    # Format dataset
    def format_prompt(example):
        if example['input']:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return prompt
    
    def tokenize_function(examples):
        prompts = [format_prompt(ex) for ex in examples]
        return tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Split into train/validation
    train_test = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./bloth_llama_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        group_by_length=True,
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Print memory estimate
    memory = bloth.estimate_memory_usage(
        model_name="7b",
        batch_size=training_args.per_device_train_batch_size,
        sequence_length=MAX_SEQ_LENGTH,
        precision="bf16"
    )
    print(f"\nEstimated memory usage:")
    print(f"  Model: {memory['model']:.2f} GB")
    print(f"  Activations: {memory['activations']:.2f} GB")
    print(f"  Optimizer: {memory['optimizer']:.2f} GB")
    print(f"  Total: {memory['total']:.2f} GB")
    
    # Train
    print("\nStarting training...")
    print("=" * 60)
    
    trainer.train()
    
    # Save model
    print("\nSaving model...")
    model.save_model("./bloth_llama_final")
    
    # Test generation
    print("\nTesting generation...")
    model.model.eval()
    
    test_prompt = "### Instruction:\nWrite a Python function to calculate fibonacci numbers.\n\n### Response:\n"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.model.device)
    
    with torch.no_grad():
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{generated_text}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
