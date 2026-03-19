"""
Bloth Example: Fine-tune LLaMA-3 with LoRA
============================================
This example shows how to use Bloth as a drop-in replacement for Unsloth.
Copy your Unsloth training script here and just change the imports.
"""

# ── Replace this:   from unsloth import FastLanguageModel
# ── With this:
import bloth
from bloth import FastModel

# ── Everything else stays the same ────────────────────────────────────────

model, tokenizer = FastModel.from_pretrained(
    model_name      = "meta-llama/Meta-Llama-3-8B",   # any HF model
    max_seq_length  = 4096,
    load_in_4bit    = True,   # QLoRA (saves ~75% VRAM vs full precision)
)

# Print what Bloth patched
bloth.print_device_info()
bloth.print_model_info(model)
bloth.print_gpu_memory()

# Apply LoRA adapters
model = FastModel.get_peft_model(
    model,
    r          = 16,
    lora_alpha = 32,
    # target_modules is auto-detected if not specified
)

# ── Use with TRL SFTTrainer (or any trainer) ──────────────────────────────
# from trl import SFTTrainer, SFTConfig
# from datasets import load_dataset
#
# dataset = load_dataset("yahma/alpaca-cleaned", split="train")
#
# trainer = SFTTrainer(
#     model          = model,
#     tokenizer      = tokenizer,
#     train_dataset  = dataset,
#     args           = SFTConfig(
#         output_dir                  = "./output",
#         num_train_epochs            = 3,
#         per_device_train_batch_size = 4,
#         gradient_accumulation_steps = 4,
#         learning_rate               = 2e-4,
#         bf16                        = True,
#         logging_steps               = 10,
#         save_steps                  = 500,
#     ),
# )
# trainer.train()

print("\n[Bloth] Training setup complete. Uncomment the SFTTrainer block to start!")
