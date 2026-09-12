# Enterprise SLM Fine-Tuning & Distillation Pipeline

Enterprise repository for Small Language Model (SLM) fine-tuning focused on domain-specific **Text-to-SQL distillation**.

---

## Overview

This repository provides an end-to-end framework for synthesizing, validating, and distilling complex enterprise relational queries into SLMs. 

### Phase 1: Synthetic Data Generation & Schema Validation
- **Complex Enterprise Schemas**: Includes multi-table joins, JSONB payload fields, foreign key constraints, and partitioned billing tables (`organizations`, `users`, `subscriptions`, `transactions`, `audit_logs`, `product_catalog`).
- **Deterministic SQL Parsing & Safety**: Parses generated PostgreSQL syntax using AST checks (`sqlglot` engine + pure-Python AST fallback), enforcing strict read-only query permissions (`SELECT` / CTE statements) and validating table/column schema references to eliminate hallucinated entity references.
- **Distillation Datasets**: Generates ShareGPT/Alpaca formatted JSONL splits (`data/train.jsonl` and `data/test.jsonl`) complete with system schema context, natural language questions, step-by-step reasoning traces (`<thought>...</thought>`), and PostgreSQL codeblocks.

### Phase 2: QLoRA Fine-Tuning Pipeline (`training/`)
- **4-Bit Memory Efficient QLoRA**: Fine-tune models like `unsloth/Meta-Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct` using `Unsloth` (or automatic fallback to HuggingFace `peft` + `bitsandbytes` + `trl` `SFTTrainer`).
- **Target Modules**: Configured for `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` with `r=16`, `lora_alpha=16`, `lora_dropout=0`, and `bias="none"`.
- **Hardware Verification**: Includes `training/dry_run.py` to inspect CUDA VRAM allocation and run verification training steps on mock data.

---

## Directory Structure

```
slm-finetuning-pipeline/
├── data/
│   ├── train.jsonl             # 40 pre-validated records (ShareGPT format)
│   └── test.jsonl              # 10 pre-validated records (ShareGPT format)
├── data_engine/
│   ├── __init__.py
│   ├── schemas.py              # PostgreSQL database schemas & context prompts
│   ├── generator.py            # Synthetic dataset generator & 50 schema seed records
│   └── validator.py            # AST & Schema SQL validator
├── training/
│   ├── __init__.py
│   ├── train.py                # 4-bit QLoRA SFT fine-tuning pipeline
│   └── dry_run.py              # Hardware VRAM verification & mock 2-step dry run
├── main.py                     # CLI entrypoint for data generation & validation
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation
```

---

## Quickstart

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. View Database Schema Context

```bash
python main.py --show-schema
```

### 3. Generate & Validate Synthetic Dataset (50 Records)

```bash
python main.py --generate
```

### 4. Fine-Tuning Dry Run (GPU Verification)

```bash
python -m training.dry_run
```

### 5. Launch QLoRA Fine-Tuning

```bash
python -m training.train \
  --model_id unsloth/Meta-Llama-3.1-8B-Instruct \
  --train_file data/train.jsonl \
  --val_file data/test.jsonl \
  --output_dir checkpoints/llama3_sql_lora \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --epochs 3 \
  --learning_rate 2e-4
```

---

## License
MIT License
