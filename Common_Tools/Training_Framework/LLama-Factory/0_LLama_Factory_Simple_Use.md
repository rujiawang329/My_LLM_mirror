# LLama Factory Simple Use
- [ENV Install](#env-install)
- [Training and Chat with LLaMA Board GUI](#training-and-chat-with-llama-board-gui)
  - [Download from ModelScope Hub](#download-from-modelscope-hub)
  - [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui)
  - [Use Fine-Tuned Model to chat with LLaMA Board GUI](#use-fine-tuned-model-to-chat-with-llama-board-gui)
  - [Pre-Training with LLaMA Board GUI](#pre-training-with-llama-board-gui)
  - [DPO Training with LLaMA Board GUI](#dpo-training-with-llama-board-gui)
- [How to use my own Dataset](#how-to-use-my-own-dataset)
  - [Copy the example dataset and change name](#copy-the-example-dataset-and-change-name)
  - [Change the name and author](#change-the-name-and-author)
  - [Update the dataset_info.json](#update-the-dataset_infojson)
  - [Start LLaMA Board GUI to fine-tuning](#start-llama-board-gui-to-fine-tuning)
  - [Start LLaMA Board GUI to chat](#start-llama-board-gui-to-chat)
- [Configure Debug Environment by cli](#configure-debug-environment-by-cli)
  - [Create a new train config](#create-a-new-train-config)
  - [Setup launch.json](#setup-launchjson)

## ENV Install
```bash
conda activate llama_factory python=3.9 -y
pip install -e ".[torch,metrics]"
```

## Training and Chat with LLaMA Board GUI

### Download from ModelScope Hub
```bash
pip install modelscope
```

```python
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='/path/to/your/desired/location')
```

### Fine-Tuning with LLaMA Board GUI
```bash
llamafactory-cli webui
```
need to set the `Model name` to `Qwen2-0.5B`\
need to set the `Model path` to the `model_dir`\
need to set the `Data` to `identity`\
need to set the `Stage` to `Supervised Fine-Tuning`
need to set other related `Train configs`

### Use Fine-Tuned Model to chat with LLaMA Board GUI

1 follow examples/inference/llama3_lora_sft.yaml to wright our model's config

1.1 copy the example config file and rename
```bash
cp examples/inference/llama3_lora_sft.yaml examples/inference/qwen_0.5b_instruct_lora_sft.yaml
```

1.2 change the model_name_or_path, adapter_name_or_path and template
```
model_name_or_path: models/Qwen/Qwen2___5-0___5B-Instruct
adapter_name_or_path: saves/Qwen2-0.5B/lora/train_2024-12-25-11-28-24
```

2 start LLaMA Board GUI to chat
```bash
llamafactory-cli webchat examples/inference/qwen_0.5b_instruct_lora_sft.yaml
```

### Pre-Training with LLaMA Board GUI
```bash
llamafactory-cli webui
```
need to set the `Model name` to `Qwen2-0.5B`\
need to set the `Model path` to the `model_dir`\
need to set the `Data` to `c4_demo`\
need to set the `Stage` to `Pre-Training`
need to set other related `Train configs`

### DPO Training with LLaMA Board GUI
```bash
llamafactory-cli webui
```
need to set the `Model name` to `Qwen2-0.5B`\
need to set the `Model path` to the `model_dir`\
need to set the `Data` to `dpo_zh_demo`\
need to set the `Stage` to `DPO`
need to set other related `Train configs`

## How to use my own Dataset
refer from data/README_zh.md
use identity.json as an example

### Copy the example dataset and change name
```bash
cp data/identity.json data/identity_loki.json
```

### change the < name > and < author >
```json
{{name}} -> loki
{{author}} -> shenxiao
```

### Update the dataset_info.json
```json
  "identity_loki": {
    "file_name": "identity_loki.json"
  },
```

### start LLaMA Board GUI to fine-tuning
```bash
llamafactory-cli webui
```
need to set the `Model name` to `Qwen2-0.5B`\
need to set the `Model path` to the `model_dir`\
need to set the `Data` to `identity_loki`\
need to set the `Stage` to `Supervised Fine-Tuning`
need to set a large epoch, such as `10`
need to set other related `Train configs`


### start LLaMA Board GUI to chat
create a new inferenceconfig file
```bash
cp examples/inference/qwen_0.5b_instruct_lora_sft.yaml examples/inference/qwen_0.5b_instruct_lora_sft_identity_loki.yaml
```

change the model_name_or_path, adapter_name_or_path and template
```yaml
model_name_or_path: models/Qwen/Qwen2___5-0___5B-Instruct
adapter_name_or_path: saves/Qwen2-0.5B/lora/train_2024-12-25-19-26-05_identity_loki
template: qwen
```

start to chat
```bash
llamafactory-cli webchat examples/inference/qwen_0.5b_instruct_lora_sft_identity_loki.yaml
```


## Configure Debug Environment by cli

### create a new train config
```bash
cp examples/train_lora/llama3_lora_sft.yaml examples/train_lora/qwen_2_0.5b_instruct_lora_sft_identity_loki.yaml
```


setup launch.json
```json
        {
            "name": "LLaMA-Factory Train Debug",
            "type": "debugpy",
            "request": "launch",
            "program": "/home/re/anaconda3/envs/llama_factory/bin/llamafactory-cli",
            "args": [
                "train",
                "examples/train_lora/qwen_2_0.5b_instruct_lora_sft_identity_loki.yaml"
            ],
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "cwd": "${workspaceFolder}"
        }
```
