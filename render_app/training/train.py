"""
training/train.py
QLoRA fine-tuning of Phi-3-mini-4k-instruct on the LIAR fact-checking dataset.

─── Prerequisites ────────────────────────────────────────────────────────────
Run this in Google Colab (T4 GPU, free tier is sufficient for 3B training).

Install:
  pip install transformers peft trl datasets bitsandbytes accelerate sentencepiece

HuggingFace login (needed for Phi-3):
  huggingface-cli login   # or: from huggingface_hub import login; login("hf_...")

Then run dataset.py first to generate data/train.jsonl etc.

─── What gets trained ────────────────────────────────────────────────────────
Base model : microsoft/Phi-3-mini-4k-instruct (3.8B, MIT license)
Method     : QLoRA — 4-bit quantized base + LoRA adapters (~80M trainable params)
Task       : Instruction-tuned JSON output for fact-check classification
Dataset    : LIAR (12.8K statements, mapped to 4 verdict classes)
Time       : ~45–60 min on T4 GPU for 3 epochs

─── Output ───────────────────────────────────────────────────────────────────
Saved locally at OUTPUT_DIR and pushed to HuggingFace as HF_REPO_ID.
"""

import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR    = "./sachbol-factcheck-3b"
HF_REPO_ID    = "prathameshbandal/sachbol-factcheck-3b"   # ← Your HF username

MAX_SEQ_LENGTH = 1024

LORA_CFG = LoraConfig(
    task_type    = TaskType.CAUSAL_LM,
    r            = 16,          # LoRA rank — higher = more capacity
    lora_alpha   = 32,          # Scaling factor (alpha/r = 2 is standard)
    lora_dropout = 0.05,
    bias         = "none",
    target_modules = [          # Phi-3 attention + MLP layers
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
)

QUANT_CFG = BitsAndBytesConfig(
    load_in_4bit             = True,
    bnb_4bit_quant_type      = "nf4",          # Normal float 4-bit
    bnb_4bit_compute_dtype   = torch.float16,
    bnb_4bit_use_double_quant = True,           # Nested quantization saves ~0.4 GB
)

TRAIN_ARGS = TrainingArguments(
    output_dir               = OUTPUT_DIR,
    num_train_epochs         = 3,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,            # Effective batch size = 16
    learning_rate            = 2e-4,
    fp16                     = True,
    optim                    = "paged_adamw_32bit",
    lr_scheduler_type        = "cosine",
    warmup_ratio             = 0.05,
    logging_steps            = 50,
    save_steps               = 250,
    eval_steps               = 250,
    evaluation_strategy      = "steps",
    save_total_limit         = 2,
    load_best_model_at_end   = True,
    metric_for_best_model    = "eval_loss",
    report_to                = "none",          # Set to "wandb" if you use W&B
    dataloader_num_workers   = 2,
    group_by_length          = True,            # Reduces padding, speeds up training
)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def apply_chat_template(sample: dict, tokenizer) -> dict:
    """Convert message list to the model's native chat format string."""
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ─── Main training function ───────────────────────────────────────────────────

def train():
    # Validate data exists
    for split in ["train", "validation"]:
        path = f"data/{split}.jsonl"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run dataset.py first: python dataset.py"
            )

    print(f"Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model with 4-bit quantization: {BASE_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config = QUANT_CFG,
        device_map          = "auto",
        trust_remote_code   = True,
        torch_dtype         = torch.float16,
        attn_implementation = "flash_attention_2",   # Remove if not supported
    )
    model.config.use_cache                  = False
    model.config.pretraining_tp             = 1
    model.gradient_checkpointing_enable()

    # Apply LoRA adapters
    model = get_peft_model(model, LORA_CFG)
    model.print_trainable_parameters()
    # Expected output: ~1-2% of total params are trainable

    # Prepare datasets
    print("Preparing datasets...")
    train_raw = load_jsonl("data/train.jsonl")
    val_raw   = load_jsonl("data/validation.jsonl")

    train_ds = Dataset.from_list(train_raw).map(
        lambda x: apply_chat_template(x, tokenizer)
    )
    val_ds = Dataset.from_list(val_raw).map(
        lambda x: apply_chat_template(x, tokenizer)
    )

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val   samples: {len(val_ds)}")

    trainer = SFTTrainer(
        model           = model,
        args            = TRAIN_ARGS,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        dataset_text_field = "text",
        max_seq_length  = MAX_SEQ_LENGTH,
        packing         = False,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nStarting QLoRA fine-tuning...")
    trainer.train()

    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Pushing to HuggingFace Hub: {HF_REPO_ID}")
    model.push_to_hub(HF_REPO_ID, private=False)
    tokenizer.push_to_hub(HF_REPO_ID)

    print(f"\nDone! Model available at: https://huggingface.co/{HF_REPO_ID}")
    print("Update LOCAL_MODEL_HF_ID in backend/config.py to match.")


if __name__ == "__main__":
    train()
