# Table of Contents
- [Megatron-LM Simple Use](#megatron-lm-simple-use)
  - [ENV](#env)
  - [DownLoad Weights](#download-weights)
  - [DownLoad Tokenizer](#download-tokenizer)
  - [Prepare Dataset](#prepare-dataset)
    - [1. Create Simple Dataset](#1-create-simple-dataset)
    - [2. Process Dataset](#2-process-dataset)
  - [Create Training Config File](#create-training-config-file)
  - [Run Train](#run-train)

# Megatron-LM Simple Use
## ENV
1. 拉取镜像
```bash
docker pull nvcr.io/nvidia/pytorch:24.12-py3
```
2. 运行镜像，并作文件映射和端口指定
```bash
docker run --shm-size=16g --gpus all -it -v /media/re/2384a6b4-4dae-400d-ad72-9b7044491b55/LLM_LR/Megatron-LM:/workspace/shenxiao -p 3333:22 nvcr.io/nvidia/pytorch:24.12-py3
```

## DownLoad Weights
```bash
mkdir Weights
cd Weights
mkdir GPT2
cd GPT2
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
unzip megatron_lm_345m_v0.0.zip
rm -r megatron_lm_345m_v0.0.zip
```

## DownLoad Tokenizer
```bash
mkdir Tokenizer
cd Tokenizer
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

## Prepare Dataset
### 1. Create Simple Dataset
```bash
mkdir Data
cd Data
mkdir GPT2_SELFDATA
cd GPT2_SELFDATA
touch data.json
echo '{"text": "Hello, world!"} {"text": "My name is Shenxiao."} {"text": "I am learning LLM."}' > data.json
```

### 2. Process Dataset
```bash
cd Megatron_Code
python tools/preprocess_data.py \
       --input /workspace/shenxiao/Data/GPT2_SELFDATA/GPT2_SELFDATA/data.json \
       --output-prefix my-gpt2 \
       --vocab-file /workspace/shenxiao/Weights/GPT2/Tokenizer/gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file /workspace/shenxiao/Weights/GPT2/Tokenizer/gpt2-merges.txt \
       --append-eod \
       --workers 4
```
```bash
mkdir /workspace/shenxiao/Processed_Data
mkdir /workspace/shenxiao/Processed_Data/GPT2_SELFDATA
mv my-gpt2* /workspace/shenxiao/Processed_Data/GPT2_SELFDATA/
```

##　Create Training Config File

创建训练日志文件夹
```bash
mkdir /workspace/shenxiao/Train_Logs
mkdir /workspace/shenxiao/Train_Logs/GPT2
```

复制训练模板
```bash
mkdir examples/gpt2
cp examples/gpt3/train_gpt3_175b_distributed.sh examples/gpt2/train_gpt2_345m_distributed.sh
```

修改训练模板
```bash
vim examples/gpt2/train_gpt2_345m_distributed.sh
```

```shell
GPUS_PER_NODE=1

CHECKPOINT_PATH=/workspace/shenxiao/Weights/GPT2 #<Specify path>
TENSORBOARD_LOGS_PATH=/workspace/shenxiao/Train_Logs/GPT2 #<Specify path>
VOCAB_FILE=/workspace/shenxiao/Weights/GPT2/Tokenizer/gpt2-vocab.json #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=/workspace/shenxiao/Weights/GPT2/Tokenizer/gpt2-merges.txt #<Specify path to file>/gpt2-merges.txt
DATA_PATH=/workspace/shenxiao/Processed_Data/GPT2_SELFDATA/my-gpt2_text_document #<Specify path and file prefix>_text_document

GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 512
    --num-attention-heads 8
    --seq-length 1024
    --max-position-embeddings 1024
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1
    # --rampup-batch-size 1 1 10 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)
```

## Run Train
```bash
bash examples/gpt2/train_gpt2_345m_distributed.sh