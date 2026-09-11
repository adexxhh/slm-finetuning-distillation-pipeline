"""
Enterprise Data Engine CLI Entrypoint.
Runs validation on synthetic SQL data, generates train/test datasets, and displays dataset statistics.
"""

import sys
import os
import argparse
import json

from data_engine.schemas import get_schema_context_prompt
from data_engine.validator import SQLValidator
from data_engine.generator import generate_seed_dataset


def validate_existing_file(filepath: str) -> None:
    """Validates an existing dataset jsonl file against sqlglot schema rules."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)

    validator = SQLValidator()
    total = 0
    passed = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            data = json.loads(line)
            # Find assistant turn
            assistant_text = ""
            for msg in data.get("conversations", []):
                if msg.get("from") == "gpt":
                    assistant_text = msg.get("value", "")
                    break
            
            # Extract SQL codeblock
            if "```sql" in assistant_text:
                sql = assistant_text.split("```sql")[1].split("```")[0].strip()
            else:
                sql = assistant_text.strip()

            is_valid, err = validator.validate(sql)
            if is_valid:
                passed += 1
            else:
                print(f"Line {line_no} Failed Validation: {err}")

    print(f"\nValidation Summary for '{filepath}':")
    print(f"Total Records: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")


def main():
    parser = argparse.ArgumentParser(description="SLM Text-to-SQL Synthetic Data & Validation Engine")
    parser.add_argument("--generate", action="store_true", help="Generate 50 validated sample records into data/ train/test splits.")
    parser.add_argument("--validate", type=str, help="Path to a .jsonl dataset file to validate.")
    parser.add_argument("--show-schema", action="store_true", help="Display the database schema prompt context.")

    args = parser.parse_args()

    if args.show_schema:
        print(get_schema_context_prompt())

    if args.validate:
        validate_existing_file(args.validate)

    if args.generate or (not args.validate and not args.show_schema):
        print("Generating Phase 1 Synthetic Text-to-SQL Dataset (50 validated records)...")
        train_p, test_p, total_cnt = generate_seed_dataset()
        print(f"Success! Generated {total_cnt} total validated records.")
        print(f"  - Train Split: {train_p}")
        print(f"  - Test Split:  {test_p}")


if __name__ == "__main__":
    main()
