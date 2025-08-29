import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# -------------------------
# Config
# -------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "data/cybersecurity_sft_dataset_300plus.jsonl"
OUTPUT_DIR = "model/cybersecurity-qwen2.5-sft"

# -------------------------
# Load Dataset
# -------------------------
dataset = load_dataset("json", data_files=DATA_PATH, split="train")


# Device/dtype (MPS prefers fp16)
if torch.backends.mps.is_available():
    device = "mps"; dtype = torch.float16
elif torch.cuda.is_available():
    device = "cuda"; major,_ = torch.cuda.get_device_capability(0); dtype = torch.bfloat16 if major >= 8 else torch.float16
else:
    device = "cpu"; dtype = torch.float32

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------------
# Model (4-bit quantization works if CUDA available)
# -------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": device},
    trust_remote_code=True,
    torch_dtype="auto",  # will auto-select float32 on MPS
)

# -------------------------
# LoRA Config
# -------------------------
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# -------------------------
# Training Config (SFTConfig)
# -------------------------
training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,                # disable on MPS
    bf16=False,                # disable if no BF16 hardware
    push_to_hub=False
)

# -------------------------
# Trainer
# -------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_config,
    peft_config=peft_config,
    formatting_func=lambda ex: f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
)

# -------------------------
# Train
# -------------------------
trainer.train()

# Save adapter + tokenizer
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
