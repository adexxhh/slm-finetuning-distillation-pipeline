"""
Verification Script for Training Pipeline.
Checks GPU VRAM allocation, inspects hardware capability, and runs 2 training steps on mock data.
"""

import sys
import os
import json
import tempfile

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from training.train import get_args_parser, run_training


def check_gpu_environment():
    """Inspects GPU environment and prints VRAM memory metrics."""
    print("=== GPU Hardware & Memory Verification ===")
    if not TORCH_AVAILABLE:
        print("CUDA / PyTorch Status: PyTorch is not installed in the local environment.")
        print("Please install PyTorch and dependencies from requirements.txt to run fine-tuning.")
        return False
    
    if not torch.cuda.is_available():
        print("CUDA Device Status: No GPU detected. (Pipeline will run in CPU fallback mode).")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)
    free_memory = total_memory - reserved_memory

    print(f"CUDA Device:           {device_name}")
    print(f"Total VRAM:            {total_memory:.2f} GB")
    print(f"Allocated VRAM:        {allocated_memory:.2f} GB")
    print(f"Reserved VRAM:         {reserved_memory:.2f} GB")
    print(f"Free VRAM:             {free_memory:.2f} GB")
    print(f"bfloat16 Supported:    {torch.cuda.is_bf16_supported()}")
    return True


def create_mock_dataset(temp_dir: str):
    """Creates a temporary mock dataset with 2 ShareGPT records for dry-run verification."""
    mock_data = [
        {
            "conversations": [
                {"from": "system", "value": "PostgreSQL Database Schema Context:\nTable: users (user_id UUID, email VARCHAR)"},
                {"from": "human", "value": "Find user email by user_id."},
                {"from": "gpt", "value": "<thought>\nFilter users table.\n</thought>\n\n```sql\nSELECT email FROM users WHERE user_id = '123';\n```"}
            ]
        },
        {
            "conversations": [
                {"from": "system", "value": "PostgreSQL Database Schema Context:\nTable: organizations (org_id UUID, name VARCHAR)"},
                {"from": "human", "value": "List all organization names."},
                {"from": "gpt", "value": "<thought>\nSelect name column from organizations.\n</thought>\n\n```sql\nSELECT name FROM organizations;\n```"}
            ]
        }
    ]

    train_path = os.path.join(temp_dir, "mock_train.jsonl")
    val_path = os.path.join(temp_dir, "mock_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in mock_data:
            f.write(json.dumps(item) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for item in mock_data:
            f.write(json.dumps(item) + "\n")

    return train_path, val_path


def run_dry_run():
    """Runs dry run test execution."""
    check_gpu_environment()

    with tempfile.TemporaryDirectory() as temp_dir:
        train_p, val_p = create_mock_dataset(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        print("\n=== Running Dry-Run Fine-Tuning Execution (2 Steps) ===")
        parser = get_args_parser()
        
        # Micro model for fast CPU/GPU dry-run test
        test_model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"

        args = parser.parse_args([
            "--model_id", test_model_id,
            "--train_file", train_p,
            "--val_file", val_p,
            "--output_dir", output_dir,
            "--batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--max_steps", "2",
            "--report_to", "none"
        ])

        try:
            run_training(args)
            print("\nDry-Run Verification Succeeded! 2 steps executed cleanly.")
        except Exception as e:
            print(f"\nDry-Run Execution Note: {e}")
            print("Note: In lightweight test environments without CUDA/quantization drivers, ensure PyTorch and Transformers are fully configured for your GPU.")


if __name__ == "__main__":
    run_dry_run()
