"""
Memory-efficient 4-bit QLoRA SFT Fine-Tuning Pipeline for Text-to-SQL Distillation.
Uses Unsloth if available, with automatic fallback to PEFT + BitsAndBytes + Trainer.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

# Lazy import ML libraries inside run_training to allow lightweight CLI --help execution
try:
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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


def format_and_tokenize(example, tokenizer, max_seq_length):
    from training.formatting import encode_completion
    return encode_completion(example, tokenizer, max_seq_length)


def run_training(args: argparse.Namespace):
    """Main training execution logic."""
    if not IMPORTS_OK:
        raise ImportError("Fine-tuning dependencies are not installed. Install requirements-ml.txt first.")
    
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
    
    if args.val_file and Path(args.train_file).resolve() == Path(args.val_file).resolve():
        raise ValueError("Training and validation paths must differ")
    train_dataset = load_jsonl_dataset(args.train_file)
    val_dataset = load_jsonl_dataset(args.val_file) if args.val_file and os.path.exists(args.val_file) else None
    if val_dataset is not None:
        from evals.benchmark import record_id
        if {record_id(x) for x in train_dataset} & {record_id(x) for x in val_dataset}:
            raise ValueError("Training/validation prompt overlap detected")
    
    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.allow_cpu:
        raise RuntimeError("4-bit QLoRA requires CUDA. Use --allow_cpu only for a small full-precision LoRA smoke model.")
    from transformers import set_seed
    set_seed(args.seed)
    device_map = "auto" if has_cuda else "cpu"
    is_bf16_supported = has_cuda and torch.cuda.is_bf16_supported()

    if args.use_unsloth and HAS_UNSLOTH and has_cuda:
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
        print("\nLoading model for " + ("4-bit QLoRA" if has_cuda else "full-precision CPU LoRA") + "...")
        if args.use_unsloth and not HAS_UNSLOTH:
            print("Notice: Unsloth not detected in environment; using standard PEFT fallback.")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if is_bf16_supported else torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if has_cuda else None

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
        
        if has_cuda:
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

    # Tokenize datasets and remove raw text columns
    tokenized_train = train_dataset.map(
        lambda ex: format_and_tokenize(ex, tokenizer, args.max_seq_length),
        batched=False,
        remove_columns=train_dataset.column_names,
    )
    
    tokenized_val = None
    if val_dataset:
        tokenized_val = val_dataset.map(
            lambda ex: format_and_tokenize(ex, tokenizer, args.max_seq_length),
            batched=False,
            remove_columns=val_dataset.column_names,
        )

    # Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        eval_strategy="epoch" if tokenized_val is not None else "no",
        save_strategy="epoch",
        max_steps=args.max_steps if hasattr(args, "max_steps") and args.max_steps else -1,
        logging_steps=1,
        fp16=has_cuda and not is_bf16_supported,
        bf16=is_bf16_supported,
        report_to=args.report_to,
        seed=args.seed,
        use_cpu=not has_cuda,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    print("\nStarting SFT Training...")
    train_result = trainer.train()
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    if tokenized_val is not None:
        trainer.save_metrics("eval", trainer.evaluate())

    print(f"\nTraining completed! Saving LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    manifest = {
        "base_model": args.model_id, "adapter_dir": args.output_dir,
        "quantized_4bit": has_cuda, "mode": "QLoRA" if has_cuda else "full-precision CPU LoRA",
        "train_examples":len(train_dataset), "validation_examples":len(val_dataset) if val_dataset else 0,
        "seed":args.seed, "epochs":args.epochs, "max_steps":args.max_steps,
        "torch":torch.__version__
    }
    Path(args.output_dir, "run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("Done!")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="4-Bit QLoRA SFT Fine-Tuning Pipeline for Text-to-SQL SLM Distillation")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--model_id", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct", help="Base model identifier from HuggingFace / Unsloth.")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl", help="Path to train JSONL dataset.")
    parser.add_argument("--val_file", type=str, default="data/validation.jsonl", help="Path to test/val JSONL dataset.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_model", help="Directory to save fine-tuned LoRA adapters.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (overrides epochs if > 0).")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--use_unsloth", action="store_true", default=False, help="Use Unsloth FastLanguageModel engine if available.")
    parser.add_argument("--report_to", type=str, default="none", help="Comma-separated logging backends: 'tensorboard', 'wandb', 'none'.")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    run_training(args)