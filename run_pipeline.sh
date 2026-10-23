#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
echo "CPU checks passed. Real training/evaluation are explicit steps; see README.md."
