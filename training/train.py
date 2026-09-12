"""
Memory-efficient 4-bit QLoRA SFT Fine-Tuning Pipeline for Text-to-SQL Distillation.
Uses Unsloth if available, with automatic fallback to PEFT + BitsAndBytes + TRL SFTTrainer.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, List

# Lazy import ML libraries inside run_training to allow lightweight CLI --help execution
try:
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False



TARGET_LORA_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]


def load_jsonl_dataset(file_path: str) -> Any:
    """Loads a ShareGPT format JSONL file into a HuggingFace Dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    return Dataset.from_list(records)


def format_sharegpt_prompt(example: Dict[str, Any], tokenizer=None) -> Dict[str, str]:
    """Formats ShareGPT conversation messages into a single training prompt string."""
    conversations = example.get("conversations", [])
    
    # Standard format: system, human, gpt
    system_msg = ""
    user_msg = ""
    assistant_msg = ""
    
    for msg in conversations:
        role = msg.get("from", "")
        content = msg.get("value", "")
        if role == "system":
            system_msg = content
        elif role == "human":
            user_msg = content
        elif role == "gpt":
            assistant_msg = content

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Generic Chat Template Fallback
        formatted_text = (
            f"<|system|>\n{system_msg}</s>\n"
            f"<|user|>\n{user_msg}</s>\n"
            f"<|assistant|>\n{assistant_msg}</s>"
        )
        
    return {"text": formatted_text}


def run_training(args: argparse.Namespace):
    """Main training execution logic."""
    if not IMPORTS_OK:
        raise ImportError("Fine-tuning dependencies (torch, transformers, peft, trl) are not installed. Install requirements.txt first.")
    
    try:
        from unsloth import FastLanguageModel
        HAS_UNSLOTH = True
    except ImportError:
        HAS_UNSLOTH = False
    print(f"=== Starting QLoRA Fine-Tuning Pipeline ===")
    print(f"Base Model ID: {args.model_id}")
    print(f"Train File:    {args.train_file}")
    print(f"Val File:      {args.val_file}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"Unsloth Engine Requested: {args.use_unsloth}")
    print(f"Unsloth Engine Available: {HAS_UNSLOTH}")
    
    # Load Datasets
    train_dataset = load_jsonl_dataset(args.train_file)
    val_dataset = load_jsonl_dataset(args.val_file) if args.val_file and os.path.exists(args.val_file) else None
    
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if args.use_unsloth and HAS_UNSLOTH:
        print("\nLoading model with Unsloth FastLanguageModel (4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=TARGET_LORA_MODULES,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
        print("\nLoading model with PEFT + Transformers + BitsAndBytes (4-bit QLoRA)...")
        if args.use_unsloth and not HAS_UNSLOTH:
            print("Notice: Unsloth not detected in environment; using standard PEFT fallback.")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if is_bf16_supported else torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if torch.cuda.is_available() else None

        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"quantization_config": bnb_config} if bnb_config else {}
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=device_map,
            trust_remote_code=True,
            **model_kwargs
        )
        
        if torch.cuda.is_available():
            model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=TARGET_LORA_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    # Format datasets
    train_dataset = train_dataset.map(lambda ex: format_sharegpt_prompt(ex, tokenizer))
    if val_dataset:
        val_dataset = val_dataset.map(lambda ex: format_sharegpt_prompt(ex, tokenizer))

    # Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        fp16=not is_bf16_supported and torch.cuda.is_available(),
        bf16=is_bf16_supported,
        logging_steps=1,
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to=args.report_to.split(",") if args.report_to != "none" else [],
        save_strategy="epoch",
        optim="adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        weight_decay=0.01,
        warmup_ratio=0.03,
        seed=3407,
    )

    # SFTTrainer Setup
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    print("\nStarting SFT Training...")
    trainer.train()

    print(f"\nTraining completed! Saving LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="4-Bit QLoRA SFT Fine-Tuning Pipeline for Text-to-SQL SLM Distillation")
    parser.add_argument("--model_id", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct", help="Base model identifier from HuggingFace / Unsloth.")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl", help="Path to train JSONL dataset.")
    parser.add_argument("--val_file", type=str, default="data/test.jsonl", help="Path to test/val JSONL dataset.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_model", help="Directory to save fine-tuned LoRA adapters.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs if > 0).")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--use_unsloth", action="store_true", default=True, help="Use Unsloth FastLanguageModel engine if available.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Comma-separated logging backends: 'tensorboard', 'wandb', 'none'.")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    run_training(args)
