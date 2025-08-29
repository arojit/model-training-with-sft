# Cybersecurity Domain-Specific SFT Model

This project fine-tunes Qwen2.5-3B-Instruct
 using Supervised Fine-Tuning (SFT) to act as a Cybersecurity Expert.
The model is trained on a curated dataset of OWASP Top 10, MITRE ATT&CK, and recent CVEs (2020–2025).

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/arojit/model-training-with-sft
cd model-training-with-sft
```

### 2. Create and Activate Virtual Environment (Recommended)

```bash
# For Linux or macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```


## Usage

### Step 1: Training
```bash
python src/sft_train_qwen.py
```
This will:

- Load Qwen2.5-3B-Instruct

- Apply LoRA adapters with QLoRA efficiency

- Fine-tune on the cybersecurity dataset

- Save the model adapters into model/cybersecurity-qwen2.5-sft

### Step 2: Inference
```bash
python src/test_sft_infer.py
```
### Example Output
```
assistant
{
  "summary": "CVE-2023-27350 is a vulnerability in the Apache Tomcat web server where an attacker can execute arbitrary code via crafted HTTP requests. This flaw lies in improper validation of input parameters.",
  "risks": [
    "Unauthorized code execution",
    "Data theft or corruption",
    "Loss of confidentiality and integrity"
  ],
  "fixes": [
    "Apply vendor patches promptly",
    "Restrict access to Tomcat via least privilege",
    "Implement input validation at application level"
  ]
}
```

## Model Behavior

The fine-tuned model acts as a Cybersecurity Expert, capable of:

- Explaining CVEs in plain language

- Suggesting mitigation strategies

- Summarizing MITRE ATT&CK techniques

- Recommending secure coding practices

- Answering OWASP Top 10 related questions
