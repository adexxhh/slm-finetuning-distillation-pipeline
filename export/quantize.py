"""Merge real adapters, optionally convert/quantize with an external llama.cpp build."""
from pathlib import Path
import argparse,subprocess
def merge_lora_16bit(base_model_id,lora_path,output_dir):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    if not Path(lora_path,"adapter_config.json").exists():
        raise FileNotFoundError("A real saved LoRA adapter is required")
    base=AutoModelForCausalLM.from_pretrained(base_model_id,local_files_only=True)
    merged=PeftModel.from_pretrained(base,lora_path,local_files_only=True).merge_and_unload()
    # CPU merge uses FP32; CUDA users can convert explicitly after numerical validation.
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    merged.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(base_model_id,local_files_only=True).save_pretrained(output_dir)
    return str(output_dir)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base_model_id",required=True,help="Downloaded local model directory")
    p.add_argument("--lora_path",required=True)
    p.add_argument("--output_dir",required=True)
    p.add_argument("--format",choices=["merged","gguf"],default="merged")
    p.add_argument("--llama_cpp_dir")
    args=p.parse_args()
    if args.format=="gguf":
        if not args.llama_cpp_dir:p.error("GGUF needs --llama_cpp_dir with converter and built llama-quantize")
        cpp=Path(args.llama_cpp_dir)
        converter=cpp/"convert_hf_to_gguf.py"
        quantizer=cpp/"build/bin/llama-quantize"
        if not quantizer.exists():quantizer=quantizer.with_suffix(".exe")
        if not converter.exists() or not quantizer.exists():p.error("Missing llama.cpp converter/binary")
    output=Path(args.output_dir)
    merged=merge_lora_16bit(args.base_model_id,args.lora_path,output/"merged")
    if args.format=="gguf":
        import sys
        fp16=output/"model-f16.gguf";quantized=output/"model-q4_k_m.gguf"
        subprocess.run([sys.executable,str(converter),merged,"--outtype","f16","--outfile",str(fp16)],check=True)
        subprocess.run([str(quantizer),str(fp16),str(quantized),"Q4_K_M"],check=True)
        if not quantized.exists():raise RuntimeError("GGUF export did not produce weights")
        (output/"Modelfile").write_text("FROM "+str(quantized.resolve())+"\n",encoding="utf-8")
    print("Exported actual weights to",output)
if __name__=="__main__":main()
