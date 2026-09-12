#!/usr/bin/env bash
# Turnkey End-to-End Pipeline Execution Script
set -e

echo "================================================================="
echo " Enterprise SLM Fine-Tuning & Text-to-SQL Distillation Pipeline"
echo "================================================================="

echo -e "\n[Phase 1/5] Synthetic Data Generation & Schema Validation..."
python3 main.py --generate

echo -e "\n[Phase 1/5] Validating Generated Datasets..."
python3 main.py --validate data/train.jsonl
python3 main.py --validate data/test.jsonl

echo -e "\n[Phase 2/5] Running Hardware & Fine-Tuning Dry-Run Verification..."
python3 -m training.dry_run

echo -e "\n[Phase 3/5] Generating Ollama Deployment Modelfile & Export Formats..."
python3 -m export.quantize --format gguf

echo -e "\n[Phase 5/5] Running Automated Comparative Evaluation Benchmark Suite..."
python3 -m evals.benchmark --num_samples 50

echo -e "\n================================================================="
echo " Turnkey Pipeline Run Complete!"
echo " Results available in evals/results/report.md"
echo " Start serving via: python -m uvicorn serving.app:app --port 8000"
echo "================================================================="
