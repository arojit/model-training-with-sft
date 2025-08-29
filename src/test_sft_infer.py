import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = os.getenv("HF_LM_ID", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER = os.getenv("SFT_OUT", "model/cybersecurity-qwen2.5-sft")


if torch.backends.mps.is_available(): 
    device="mps"
    dtype=torch.float16
elif torch.cuda.is_available():
    device="cuda"
    dtype=torch.float16
else: 
    device="cpu"
    dtype=torch.float32

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=dtype,
    device_map={"": device},
    attn_implementation="eager"
)
model = PeftModel.from_pretrained(base, ADAPTER)

messages = [
    {"role":"system","content":"You are a strict JSON generator that always outputs the schema {summary, risks, fixes}."},
    {"role":"user","content":"Explain CVE-2023-27350"}
]

prompt = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tok(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False, temperature=0.0, pad_token_id=tok.eos_token_id)

print(tok.decode(out[0], skip_special_tokens=True).split(prompt)[-1].strip())