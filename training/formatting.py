"""Shared training/inference prompt format and completion-only loss."""
def prompt_messages(record):
    messages=[]
    answer=None
    for turn in record.get("conversations", []):
        role=turn.get("from")
        if role=="system": messages.append({"role":"system","content":turn["value"]})
        elif role=="human": messages.append({"role":"user","content":turn["value"]})
        elif role=="gpt": answer=turn["value"]
    if not messages or answer is None:
        raise ValueError("Record needs a prompt and assistant answer")
    return messages, answer

def encode_completion(record, tokenizer, max_seq_length):
    messages, answer=prompt_messages(record)
    prompt=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
    # Encode separately so the loss boundary cannot shift across a BPE merge.
    prefix=tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion=tokenizer(answer + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    if not completion or len(prefix)+len(completion)>max_seq_length:
        raise ValueError("Example exceeds max_seq_length or has no supervised completion; do not silently truncate")
    return {"input_ids":prefix+completion, "attention_mask":[1]*(len(prefix)+len(completion)),
            "labels":[-100]*len(prefix)+completion}
