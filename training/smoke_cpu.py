"""Offline tiny-model LoRA smoke run. This is NOT 8B QLoRA or evidence of SQL quality."""
from pathlib import Path
import argparse,json
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast,set_seed
from training.train import get_args_parser,run_training
from training.formatting import encode_completion
from serving.backend import HFBackend

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="artifacts/cpu_smoke")
    options=parser.parse_args()
    torch.set_num_threads(2);set_seed(3407)
    root=Path(options.output);base=root/"base";base.mkdir(parents=True,exist_ok=True)
    texts=["system user assistant : Return SQL only . users user_id full_name SELECT COUNT FROM ; 1 2 sql"]
    tok=Tokenizer(WordLevel(unk_token="[UNK]"));tok.pre_tokenizer=Whitespace()
    tok.train_from_iterator(texts,WordLevelTrainer(special_tokens=["[PAD]","[UNK]","[EOS]"]))
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token="[PAD]",unk_token="[UNK]",eos_token="[EOS]")
    tokenizer.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    tokenizer.save_pretrained(base)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer),hidden_size=32,
        intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,
        max_position_embeddings=256,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id))
    model.save_pretrained(base)
    records=[{"conversations":[{"from":"system","value":"Return SQL only. users user_id"},
              {"from":"human","value":q},{"from":"gpt","value":a}]} for q,a in
             [("COUNT users","SELECT COUNT(user_id) FROM users;"),("users","SELECT user_id FROM users;")]]
    encoded=encode_completion(records[0],tokenizer,256)
    assert -100 in encoded["labels"] and any(x!=-100 for x in encoded["labels"])
    train=root/"train.jsonl";validation=root/"validation.jsonl"
    train.write_text("\n".join(json.dumps(x) for x in records),encoding="utf-8")
    held_out={"conversations":[{"from":"system","value":"Return SQL only. users user_id"},
                 {"from":"human","value":"SELECT 1"},{"from":"gpt","value":"SELECT 1;"}]}
    validation.write_text(json.dumps(held_out),encoding="utf-8")
    args=get_args_parser().parse_args(["--model_id",str(base),"--train_file",str(train),
        "--val_file",str(validation),"--output_dir",str(root/"adapter"),"--allow_cpu",
        "--max_steps","2","--batch_size","1","--gradient_accumulation_steps","1",
        "--max_seq_length","256","--report_to","none"])
    run_training(args)
    backend=HFBackend(str(base),str(root/"adapter"))
    generated=backend.generate("users","Return SQL only. users user_id",max_tokens=8)
    result={"status":"passed","scope":"tiny random Llama, two CPU LoRA optimizer steps, adapter save/reload and real generation",
            "is_8b_qlora":False,"quality_evaluated":False,"generated_tokens":generated["total_tokens"],
            "generation_ms":generated["generation_ms"],"torch":torch.__version__}
    (root/"smoke_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
