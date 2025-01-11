# LLama-Factory Eval Models on Own Dataset
- [LLama-Factory Eval Models on Own Dataset](#llama-factory-eval-models-on-own-dataset)
  - [Download the Models](#download-the-models)
  - [Prepare Eval Dataset](#prepare-eval-dataset)
    - [1. Get Eval Dataset](#1-get-eval-dataset)
    - [2. Register Eval Dataset](#2-register-eval-dataset)
  - [Use llamafactory-cli to Eval](#use-llamafactory-cli-to-eval)
    - [1. Copy Eval Script](#1-copy-eval-script)
    - [2. Run Eval](#2-run-eval)
  - [Use vLLM to Eval](#use-vllm-to-eval)
    - [1. ENV: Install vLLM](#1-env-install-vllm)
    - [2. Usage Template](#2-usage-template)
    - [3. Run](#3-run)
  - [use official code completion example](#use-official-code-completion-example)
  - [Eval BLEU and Rouge](#eval-bleu-and-rouge)
    - [1. ENV](#1-env)
    - [2. create eval python script](#2-create-eval-python-script)
  - [Eval Results](#eval-results)
    - [Albation Study](#albation-study)

## Download the Models
```bash

modelscope download Qwen/Qwen2.5-Coder-0.5B --local_dir Qwen2.5-Coder-0.5B
modelscope download Qwen/Qwen2.5-Coder-3B --local_dir Qwen2.5-Coder-3B
modelscope download Qwen/Qwen2.5-Coder-7B --local_dir Qwen2.5-Coder-7B
modelscope download Qwen/Qwen2.5-Coder-0.5B-Instruct --local_dir Qwen2.5-Coder-0.5B-Instruct
modelscope download Qwen/Qwen2.5-Coder-3B-Instruct --local_dir Qwen2.5-Coder-3B-Instruct
modelscope download Qwen/Qwen2.5-Coder-7B-Instruct --local_dir Qwen2.5-Coder-7B-Instruct

huggingface-cli download Qwen/Qwen2.5-Coder-0.5B --local-dir Qwen2.5-Coder-0.5B
huggingface-cli download Qwen/Qwen2.5-Coder-3B --local-dir Qwen2.5-Coder-3B
huggingface-cli download Qwen/Qwen2.5-Coder-7B --local-dir Qwen2.5-Coder-7B
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct --local-dir Qwen2.5-Coder-0.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct --local-dir Qwen2.5-Coder-3B-Instruct
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir Qwen2.5-Coder-7B-Instruct
```

## Prepare Eval Dataset
### 1. Get Eval Dataset
```
verilog_completion_test_data_format1.json
```
```
{"instruction": "请帮我补全verilog代码", "input": "  nand2  g0297(.a(new_n388_), .b(new_n286_), .O(new_n389_));\n  inv1   g0298(.a(x44), .O(new_n390_)); \n -------- \n  nand3  g0295(.a(new_n386_), .b(new_n122_), .c(x29), .O(new_n387_));\n :", "output": "  nor2   g0296(.a(x41), .b(x40), .O(new_n388_));\n"}
```
```
verilog_completion_test_data_format2.json
```
```
{"instruction": "Only output the code completion results", "input": "  nand2  g0297(.a(new_n388_), .b(new_n286_), .O(new_n389_));\n  inv1   g0298(.a(x44), .O(new_n390_)); \n -------- \n  nand3  g0295(.a(new_n386_), .b(new_n122_), .c(x29), .O(new_n387_));\n :", "output": "  nor2   g0296(.a(x41), .b(x40), .O(new_n388_));\n"}
```


### 2. Register Eval Dataset
```
data/dataset_info.json
```
```json
  "verilog_completion_test_dataset_format1": {
    "file_name": "verilog_completion_test_data_format1.json"
  },
```
```json
  "verilog_completion_test_dataset_format2": {
    "file_name": "verilog_completion_test_data_format2.json"
  },
```

## Use llamafactory-cli to Eval
### 1. Copy Eval Script
```bash
cp examples/extras/nlg_eval/llama3_lora_predict.yaml examples/extras/nlg_eval/qwen2.5_coder_predict.yaml
```
修改参数配置
```yaml
### model
model_name_or_path: models/Qwen2.5_Coder/Qwen2.5-Coder-3B-Instruct
# adapter_name_or_path
### method
stage: sft
finetuning_type: full
### dataset
eval_dataset: verilog_completion_test_dataset_format1
template: qwen
cutoff_len: 2048
# max_samples
### output
output_dir: saves/Qwen2.5_Coder/Qwen2.5-Coder-3B-Instruct
```

### 2. Run Eval
```bash
export PYTHONPATH=$PYTHONPATH:src
```
```bash
llamafactory-cli train examples/extras/nlg_eval/qwen2.5_coder_predict.yaml
```

**Too Slow, Use vLLM**

## Use vLLM to Eval

### 1. ENV: Install vLLM
```
llamafactory 0.9.2.dev0
```
```bash
pip install vllm==0.6.6.post1
```

### 2. Usage Template
```
scripts/vllm_infer.py
python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
```

### 3. Run Eval
```bash

python scripts/vllm_infer.py --model_name_or_path models/Qwen2.5_Coder/Qwen2.5-Coder-0.5B-Instruct --template qwen --dataset verilog_completion_test_dataset_format3
python scripts/vllm_infer.py --model_name_or_path models/Qwen2.5_Coder/Qwen2.5-Coder-3B-Instruct --template qwen --dataset verilog_completion_test_dataset_format3
python scripts/vllm_infer.py --model_name_or_path models/Qwen2.5_Coder/Qwen2.5-Coder-7B-Instruct --template qwen --dataset verilog_completion_test_dataset_format3

python scripts/vllm_infer.py --model_name_or_path models/Qwen2.5_Coder/Qwen2.5-Coder-0.5B --template qwen --dataset verilog_completion_test_dataset_format3
python scripts/vllm_infer.py --model_name_or_path models/Qwen2.5_Coder/Qwen2.5-Coder-3B --template qwen --dataset verilog_completion_test_dataset_format3
python scripts/vllm_infer.py --model_name_or_path models/Qwen2.5_Coder/Qwen2.5-Coder-7B --template qwen --dataset verilog_completion_test_dataset_format3
```

## use official code completion example
File-Level Code Completion (Fill in the middle)
```
prompt = '<|fim_prefix|>' + prefix_code + '<|fim_suffix|>' + suffix_code + '<|fim_middle|>'
```
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
# load model
device = "cuda:0" # the device to load the model onto

TOKENIZER = AutoTokenizer.from_pretrained("models/Qwen2.5_Coder/Qwen2.5-Coder-0.5B-Instruct")
MODEL = AutoModelForCausalLM.from_pretrained("models/Qwen2.5_Coder/Qwen2.5-Coder-0.5B-Instruct", device_map="auto").eval()

input_text = """<|fim_prefix|>def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    <|fim_suffix|>
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)<|fim_middle|>"""

model_inputs = TOKENIZER([input_text], return_tensors="pt").to(device)

# Use `max_new_tokens` to control the maximum output length.
generated_ids = MODEL.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=False)[0]
# The generated_ids include prompt_ids, we only need to decode the tokens after prompt_ids.
output_text = TOKENIZER.decode(generated_ids[len(model_inputs.input_ids[0]):], skip_special_tokens=True)

print(f"Prompt: {input_text}\n\nGenerated text: {output_text}")
```

## Eval BLEU and Rouge

### 1. ENV
```bash
pip install rouge_score
pip install nltk
```

### 2. create eval python script
```python
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json
from tqdm import tqdm

def tokenize(text):
    # Split by whitespace and filter out empty tokens
    return [token for token in text.split() if token.strip()]

def compute_metrics_for_pair(pred, label):
    score_dict = {
        "rouge-1": 0,
        "rouge-2": 0,
        "rouge-l": 0,
        "bleu-4": 0
    }
    
    # Tokenize prediction and reference
    hypothesis = tokenize(pred)
    reference = tokenize(label)
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    scores = scorer.score(' '.join(reference), ' '.join(hypothesis))
    
    score_dict["rouge-1"] = round(scores['rouge1'].fmeasure * 100, 4)
    score_dict["rouge-2"] = round(scores['rouge2'].fmeasure * 100, 4)
    score_dict["rouge-l"] = round(scores['rougeLsum'].fmeasure * 100, 4)
    
    # Calculate BLEU score
    bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
    score_dict["bleu-4"] = round(bleu_score * 100, 4)
    
    return score_dict

def evaluate_predictions(jsonl_path):
    total_scores = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    
    # Read and evaluate each line from the JSONL file
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Evaluating"):
        data = json.loads(line)
        pred = data['predict']
        label = data['label']
        
        scores = compute_metrics_for_pair(pred, label)
        for metric, score in scores.items():
            total_scores[metric].append(score)
    
    # Calculate mean scores
    final_scores = {
        metric: float(np.mean(scores)) 
        for metric, scores in total_scores.items()
    }
    
    return final_scores

if __name__ == "__main__":
    jsonl_path = "./saves/Qwen2.5_Coder/Qwen2.5-Coder-0.5B-Instruct_format2/generated_predictions.jsonl"  # Update this path if needed
    results = evaluate_predictions(jsonl_path)
    
    print("\nEvaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.2f}")
```

## Eval Results
| Model Name                      | w or w/o Instruction | BLEU-4 | Rouge-1 | Rouge-2 | Rouge-l |
| ------------------------------- | -------------------- | ------ | ------- | ------- | ------- |
| Qwen2.5_Coder_0.5B              | without              | 0.12   | 2.63    | 1.30    | 2.34    |
| Qwen2.5_Coder_0.5B              | with                 | 0.23   | 5.00    | 2.42    | 4.35    |
| **Qwen2.5_Coder_3B**            | without              | 0.50   | 8.93    | 4.30    | 7.82    |
| Qwen2.5_Coder_3B                | with                 | 0.59   | 9.78    | 4.69    | 9.78    |
| Qwen2.5_Coder_7B                | without              | 0.58   | 8.61    | 4.29    | 7.45    |
| Qwen2.5_Coder_7B                | with                 | 0.57   | 9.09    | 4.57    | 7.83    |
| Qwen2.5_Coder_0.5B_Instruct     | without              | 0.25   | 5.95    | 2.61    | 4.92    |
| **Qwen2.5_Coder_0.5B_Instruct** | with                 | 1.05   | 16.96   | 8.06    | 14.24   |
| Qwen2.5_Coder_3B_Instruct       | without              | 0.25   | 6.87    | 2.64    | 5.43    |
| Qwen2.5_Coder_3B_Instruct       | with                 | 0.70   | 9.97    | 4.69    | 8.67    |
| Qwen2.5_Coder_7B_Instruct       | without              | 0.46   | 3.09    | 6.31    | 0.46    |
| **Qwen2.5_Coder_7B_Instruct**   | with                 | 1.54   | 19.33   | 10.09   | 16.72   |

### Albation Study

#### 1. Ablation Instruction

Instruction可以始终有效的较大幅度提升性能

尤其对于Instruct后的模型来说，性能提升更加明显。但对于Base模型，性能提升则不明显。

如果对于Instruct模型，不用Instruct，会比Base模型还要差。

#### 2. Ablation Instruct Version OR Base Version

Qwen2.5_Coder的Instruct相比于Base性能提升明显

Qwen2.5_Coder_Instruct_3B 训坏掉了，性能很差。

Qwen2.5_Coder_Instruct_0.5B 非常强，可以作为一个好用的小模型来玩。