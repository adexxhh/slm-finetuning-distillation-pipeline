# Enterprise SLM Fine-Tuning & Distillation Pipeline

Enterprise repository for Small Language Model (SLM) fine-tuning focused on domain-specific **Text-to-SQL distillation**.

---

## Overview

This repository provides an end-to-end framework for synthesizing, validating, and distilling complex enterprise relational queries into SLMs. 

### Phase 1: Synthetic Data Generation & Schema Validation
- **Complex Enterprise Schemas**: Includes multi-table joins, JSONB payload fields, foreign key constraints, and partitioned billing tables (`organizations`, `users`, `subscriptions`, `transactions`, `audit_logs`, `product_catalog`).
- **Deterministic SQL Parsing & Safety**: Parses generated PostgreSQL syntax using AST checks (`sqlglot` engine + pure-Python AST fallback), enforcing strict read-only query permissions (`SELECT` / CTE statements) and validating table/column schema references to eliminate hallucinated entity references.
- **Distillation Datasets**: Generates ShareGPT/Alpaca formatted JSONL splits (`data/train.jsonl` and `data/test.jsonl`) complete with system schema context, natural language questions, step-by-step reasoning traces (`<thought>...</thought>`), and PostgreSQL codeblocks.

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

Outputs:
- `data/train.jsonl` (40 validated samples)
- `data/test.jsonl` (10 validated samples)

### 4. Validate Custom Dataset

```bash
python main.py --validate data/train.jsonl
```

---

## License
MIT License
