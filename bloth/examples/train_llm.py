"""
Bloth v1.0 — Training Example
================================
Fine-tune any LLM with Bloth kernels.
Designed to run in Google Colab with a T4/A100/L4 GPU.

Step 1: Install
    !git clone https://github.com/shell-bay/bloth.git
    %cd bloth
    !pip install -e .

Step 2: Run this script
    python examples/train_llm.py
"""

# ── Imports ────────────────────────────────────────────────────────────────
from bloth import FastModel, print_device_info, print_gpu_memory

# Unsloth users: just change the import above — everything below stays the same!

print_device_info()

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME     = "unsloth/Llama-3.2-1B"   # small model, good for testing
MAX_SEQ_LENGTH = 2048
LORA_RANK      = 16
BATCH_SIZE     = 2
GRAD_ACCUM     = 4
LR             = 2e-4
MAX_STEPS      = 60
OUTPUT_DIR     = "./bloth_output"

# ── Load model ─────────────────────────────────────────────────────────────
model, tokenizer = FastModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = True,     # QLoRA — uses ~75% less VRAM
    trust_remote_code=True,
)

print_gpu_memory()

# ── Apply LoRA ─────────────────────────────────────────────────────────────
model = FastModel.get_peft_model(
    model,
    r              = LORA_RANK,
    lora_alpha     = LORA_RANK * 2,
    lora_dropout   = 0.05,
)

# ── Dataset ────────────────────────────────────────────────────────────────
from datasets import load_dataset

dataset = load_dataset("yahma/alpaca-cleaned", split="train")

def format_prompt(sample):
    if sample.get("input"):
        return (f"### Instruction:\n{sample['instruction']}\n\n"
                f"### Input:\n{sample['input']}\n\n"
                f"### Response:\n{sample['output']}")
    return (f"### Instruction:\n{sample['instruction']}\n\n"
            f"### Response:\n{sample['output']}")

# ── Trainer ────────────────────────────────────────────────────────────────
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args          = SFTConfig(
        max_seq_length              = MAX_SEQ_LENGTH,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps                = 10,
        max_steps                   = MAX_STEPS,
        learning_rate               = LR,
        fp16                        = False,
        bf16                        = True,
        logging_steps               = 1,
        output_dir                  = OUTPUT_DIR,
        optim                       = "adamw_8bit",
        seed                        = 42,
    ),
)

print("\n[Bloth] Starting training...")
print_gpu_memory()
trainer.train()
print("\n[Bloth] Training complete! ✅")
print_gpu_memory()
