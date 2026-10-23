"""Real local Hugging Face inference. No model is downloaded by default."""
import os
import re
import time

def extract_sql(text):
    match=re.search(r"```sql\s*(.*?)```",text,re.S|re.I)
    return match.group(1).strip() if match else text.strip()

class HFBackend:
    name="huggingface"
    def __init__(self, model_path, adapter_path=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        kwargs={}
        if torch.cuda.is_available():
            kwargs={"device_map":"auto","quantization_config":BitsAndBytesConfig(
                load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)}
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,**kwargs)
        if adapter_path:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,adapter_path,local_files_only=True)
        self.model.config.use_cache=True
        self.model.eval()
        self.tokenizer.pad_token=self.tokenizer.pad_token or self.tokenizer.eos_token

    def generate(self, question, schema, max_tokens=256):
        import torch
        messages=[{"role":"system","content":schema},{"role":"user","content":question}]
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoded=self.tokenizer(prompt,add_special_tokens=False,return_tensors="pt").to(self.model.device)
        start=time.perf_counter()
        with torch.inference_mode():
            output=self.model.generate(**encoded,max_new_tokens=max_tokens,do_sample=False,
                                       pad_token_id=self.tokenizer.pad_token_id)
        generated=output[0,encoded["input_ids"].shape[1]:]
        text=self.tokenizer.decode(generated,skip_special_tokens=True)
        elapsed=(time.perf_counter()-start)*1000
        return {"sql":extract_sql(text),"text":text,"total_tokens":int(generated.numel()),
                "generation_ms":elapsed,"ttft_ms":None,"backend":self.name}

def load_backend():
    path=os.getenv("MODEL_PATH")
    return HFBackend(path,os.getenv("ADAPTER_PATH")) if path else None
