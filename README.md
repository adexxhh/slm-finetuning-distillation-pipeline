# Enterprise SLM Fine-Tuning & Distillation Pipeline

Enterprise repository for Small Language Model (SLM) fine-tuning focused on domain-specific **Text-to-SQL distillation**.

---

## Overview

This repository provides an end-to-end framework for synthesizing, validating, fine-tuning, and exporting complex enterprise relational query SLMs.

### Phase 1: Synthetic Data Generation & Schema Validation
- **Complex Enterprise Schemas**: Includes multi-table joins, JSONB payload fields, foreign key constraints, and partitioned billing tables (`organizations`, `users`, `subscriptions`, `transactions`, `audit_logs`, `product_catalog`).
- **Deterministic SQL Parsing & Safety**: Parses generated PostgreSQL syntax using AST checks (`sqlglot` engine + pure-Python AST fallback), enforcing strict read-only query permissions (`SELECT` / CTE statements) and validating table/column schema references.
- **Distillation Datasets**: Generates ShareGPT/Alpaca formatted JSONL splits (`data/train.jsonl` and `data/test.jsonl`).

### Phase 2: QLoRA Fine-Tuning Pipeline (`training/`)
- **4-Bit Memory Efficient QLoRA**: Fine-tune models like `unsloth/Meta-Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct` using `Unsloth` (or fallback to HuggingFace `peft` + `bitsandbytes` + `trl` `SFTTrainer`).
- **Target Modules**: Configured for `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` with `r=16`, `lora_alpha=16`, `lora_dropout=0`, and `bias="none"`.
- **Hardware Verification**: Includes `training/dry_run.py` to inspect CUDA VRAM allocation and run verification training steps on mock data.

### Phase 3: Model Export & Quantization Pipeline (`export/`)
- **16-Bit LoRA Weight Merging**: Merges fine-tuned LoRA weights back into 16-bit FP16 base model weights (`export/quantize.py --format merged-16bit`).
- **GGUF Quantization (`Q4_K_M`, `Q8_0`)**: Quantizes models for edge deployment with `llama.cpp` and Ollama (`export/quantize.py --format gguf`).
- **AWQ High-Throughput Serving**: Export 4-bit AWQ quantized models for vLLM serving (`export/quantize.py --format awq`).
- **Ollama Modelfile Generation**: Auto-generates an enterprise-grade `Modelfile` pre-loaded with schema context prompts and parameters for instant deployment via `ollama create sql-slm -f ./export/Modelfile`.

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
├── export/
│   ├── __init__.py
│   ├── quantize.py             # Export & Quantization CLI (16bit, GGUF, AWQ)
│   └── Modelfile               # Auto-generated template for local Ollama deployment
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

### 2. Synthetic Dataset Generation & Validation

```bash
python main.py --generate
python main.py --validate data/train.jsonl
```

### 3. Fine-Tuning & Hardware Verification

```bash
# GPU Dry Run Verification
python -m training.dry_run

# QLoRA Training
python -m training.train \
  --model_id unsloth/Meta-Llama-3.1-8B-Instruct \
  --train_file data/train.jsonl \
  --val_file data/test.jsonl \
  --output_dir checkpoints/llama3_sql_lora
```

### 4. Export & Ollama Deployment

```bash
# Export all formats (Merged FP16, GGUF Q4_K_M/Q8_0, AWQ, Modelfile)
python -m export.quantize --format all

# Deploy fine-tuned model into local Ollama
ollama create sql-slm -f ./export/Modelfile

# Run query in Ollama
ollama run sql-slm "List all active enterprise users and their organization names."
```

---

## License
MIT License
