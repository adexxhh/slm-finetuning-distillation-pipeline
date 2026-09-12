"""
Model Export and Quantization Pipeline for Text-to-SQL Distilled SLMs.
Supports:
1. Merging LoRA weights into 16-bit FP16 base model.
2. GGUF quantization (Q4_K_M and Q8_0) for llama.cpp / Ollama edge serving.
3. AWQ / 4-bit quantization export for vLLM high-throughput serving.
4. Auto-generating dynamic Ollama Modelfile with enterprise schema system prompt.
"""

import os
import sys
import argparse
from typing import Optional

from data_engine.schemas import get_schema_context_prompt

# Lazy import ML dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    IMPORTS_OK = True
except ImportError:
    IMPORTS_OK = False

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False


def generate_ollama_modelfile(gguf_model_path: str, output_modelfile_path: str) -> str:
    """Generates an enterprise Ollama Modelfile pre-configured with schema prompt and parameters."""
    schema_prompt = get_schema_context_prompt()
    
    # Escape quotes and formatting for Modelfile
    formatted_system_prompt = schema_prompt.replace('"', '\\"')
    
    modelfile_content = f"""# Ollama Modelfile for Domain-Specific Enterprise Text-to-SQL SLM
FROM {gguf_model_path}

# Set inference parameters
PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "</s>"

# Pre-load Enterprise PostgreSQL Database Schema Prompt Context
SYSTEM \"\"\"{formatted_system_prompt}\"\"\"

# Define chat template structure
TEMPLATE \"\"\"{{{{ if .System }}}}<|system|>
{{{{ .System }}}}</s>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}}}</s>
{{{{ end }}}}<|assistant|>
{{{{ .Response }}}}</s>\"\"\"
"""
    
    os.makedirs(os.path.dirname(os.path.abspath(output_modelfile_path)), exist_ok=True)
    with open(output_modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
        
    print(f"Generated Ollama Modelfile at: {output_modelfile_path}")
    print(f"To deploy in Ollama run:\n  ollama create sql-slm -f {output_modelfile_path}")
    return modelfile_content


def merge_lora_16bit(base_model_id: str, lora_path: str, output_dir: str):
    """Merges LoRA adapter weights into 16-bit FP16 base model and saves merged checkpoint."""
    if not IMPORTS_OK:
        raise ImportError("Required ML packages (torch, transformers, peft) are not installed.")
        
    print(f"\n=== Merging LoRA Weights into 16-bit Base Model ===")
    print(f"Base Model:  {base_model_id}")
    print(f"LoRA Path:   {lora_path}")
    print(f"Output Dir:  {output_dir}")

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    print("Loading PEFT LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging weights via merge_and_unload()...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged 16-bit model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("16-bit model merge completed successfully!")
    return output_dir


def export_gguf(base_model_id: str, lora_path: str, output_dir: str, quant_methods: list):
    """Exports model to GGUF formats (Q4_K_M, Q8_0) for llama.cpp and Ollama edge inference."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== Exporting GGUF Quantized Checkpoints ({', '.join(quant_methods)}) ===")

    if HAS_UNSLOTH:
        print("Using Unsloth native FastLanguageModel GGUF export engine...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path if os.path.exists(lora_path) else base_model_id,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        for quant in quant_methods:
            out_file = os.path.join(output_dir, f"model_{quant}.gguf")
            print(f"Saving GGUF [{quant}] -> {out_file}...")
            model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=quant)
    else:
        print("Standard PEFT mode: Merging model weights prior to GGUF conversion...")
        merged_dir = os.path.join(output_dir, "merged_fp16")
        try:
            merge_lora_16bit(base_model_id, lora_path, merged_dir)
            print(f"Merged FP16 weights ready at {merged_dir}.")
        except Exception as e:
            print(f"Notice: Live 16-bit weight merge skipped ({e}).")
        print("Notice: To complete GGUF conversion, run llama.cpp convert script:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {merged_dir} --outtype q4_k_m --outfile {output_dir}/model_q4_k_m.gguf")

    gguf_primary = os.path.join(output_dir, "model_q4_k_m.gguf")
    modelfile_path = os.path.join(output_dir, "Modelfile")
    generate_ollama_modelfile(gguf_primary, modelfile_path)


def export_awq(merged_model_dir: str, output_dir: str):
    """Exports 4-bit AWQ quantized checkpoint for high-throughput serving via vLLM."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== Exporting 4-Bit AWQ Quantized Model for vLLM ===")
    print(f"Source Model: {merged_model_dir}")
    print(f"Output Dir:   {output_dir}")

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"
        }

        print("Loading model for AWQ quantization...")
        model = AutoAWQForCausalLM.from_pretrained(merged_model_dir, **{"low_cpu_mem_usage": True})
        tokenizer = AutoTokenizer.from_pretrained(merged_model_dir, trust_remote_code=True)

        print("Quantizing weights with AWQ (W4A16)...")
        model.quantize(tokenizer, quant_config=quant_config)

        print(f"Saving AWQ model to {output_dir}...")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("AWQ export completed successfully!")
    except ImportError:
        print("AutoAWQ is not installed. To export AWQ models, install autoawq via `pip install autoawq`.")
        print(f"Alternatively, save merged 16-bit weights to {output_dir} for vLLM on-the-fly quantization.")


def main():
    parser = argparse.ArgumentParser(description="Model Export & Quantization Pipeline (GGUF, AWQ, 16-bit Merge, Ollama)")
    parser.add_argument("--base_model_id", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct", help="Base model identifier.")
    parser.add_argument("--lora_path", type=str, default="checkpoints/lora_model", help="Path to fine-tuned LoRA checkpoint.")
    parser.add_argument("--output_dir", type=str, default="export", help="Output directory for exported checkpoints.")
    parser.add_argument("--format", type=str, choices=["gguf", "awq", "merged-16bit", "all"], default="all", help="Target export format.")
    parser.add_argument("--quant_types", type=str, default="q4_k_m,q8_0", help="Comma-separated GGUF quantization formats.")

    args = parser.parse_args()
    quant_methods = [q.strip() for q in args.quant_types.split(",")]

    print("=========================================================")
    print(" Enterprise SLM Model Export & Quantization Pipeline")
    print("=========================================================")

    if args.format in ["merged-16bit", "all"]:
        merged_out = os.path.join(args.output_dir, "merged_fp16")
        if IMPORTS_OK and os.path.exists(args.lora_path):
            merge_lora_16bit(args.base_model_id, args.lora_path, merged_out)
        else:
            print(f"Note: Skipping live 16-bit merge (LoRA path '{args.lora_path}' or dependencies not present).")

    if args.format in ["gguf", "all"]:
        gguf_out = os.path.join(args.output_dir, "gguf")
        export_gguf(args.base_model_id, args.lora_path, gguf_out, quant_methods)

    if args.format in ["awq", "all"]:
        awq_out = os.path.join(args.output_dir, "awq")
        merged_source = os.path.join(args.output_dir, "merged_fp16")
        export_awq(merged_source, awq_out)

    # Always ensure a default Modelfile is present in export root
    modelfile_root = os.path.join(args.output_dir, "Modelfile")
    generate_ollama_modelfile("./export/gguf/model_q4_k_m.gguf", modelfile_root)


if __name__ == "__main__":
    main()
