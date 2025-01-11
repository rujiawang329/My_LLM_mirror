# My_LLM

## Update Log

2025.01.11
docs: add PIP installation guide for Pai-Megatron-Patch
[[link]](Common_Tools/Training_Framework/Pai-Megatron-Patch/1_Use_PIP_to_Install_Pai-Megatron-Patch.md)

- Document basic environment requirements (CUDA 12.4, Python 3.10)
- Add conda environment setup instructions
- List all required dependencies with specific versions:
  - PyTorch 2.4.0
  - APEX with CUDA extensions
  - Transformers 4.44.0
  - Flash Attention v2.6.3
  - DeepSpeed 0.14.4
  - Other essential packages and system dependencies

2025.01.11
docs: add Pai-Megatron-Patch usage guide
[[link]](Common_Tools/Training_Framework/Pai-Megatron-Patch/0_Pai-Megatron-Patch_Simple_Use.md)

- Add comprehensive documentation for Pai-Megatron-Patch usage, including:
  - Environment setup with Docker
  - SSH connection instructions
  - Qwen2 model and dataset download steps
  - Model format conversion guide
  - Training procedures for PT and SFT
  - Evaluation process

2025.01.11
docs: add Megatron-LM basic usage guide
[[link]](Common_Tools/Training_Framework/Megatron-LM/0_Megatron-LM_Simple_Use.md)

- Add Docker environment setup instructions
- Include steps for downloading model weights and tokenizer
- Document dataset preparation and processing
- Add training configuration setup
- Include instructions for running GPT-2 training


2025.01.11
docs: use LLama-Factory to eval Qwen2.5-Coder models on own Verilog test datasets
[[link]](Common_Tools/Training_Framework/LLama-Factory/1_LLama_Factory_Eval_Models_on_Own_Dataset.md)

- Document model evaluation workflow using LLaMA-Factory and vLLM
- Add BLEU and ROUGE evaluation scripts
- Include comprehensive evaluation results
- Add ablation studies comparing instruction vs non-instruction models
- Compare performance across different model sizes (0.5B, 3B, 7B)

2025.01.11 
docs: add LLama-Factory basic usage guide
[[link]](Common_Tools/Training_Framework/LLama-Factory/0_LLama_Factory_Simple_Use.md)

- Add environment setup instructions
- Document training workflow with LLaMA Board GUI
- Include instructions for fine-tuning, pre-training and DPO
- Add custom dataset usage guide
- Include debug environment setup

2025.01.11
docs: add Docker common configuration guide
[[link]](Common_Tools/Linux/0_Basic_Linux_Command.md)

- Add comprehensive Docker setup and usage guide
- Include common Docker commands and installation steps
- Document SSH service configuration and complex run settings
- Add solutions for common issues (SSH, VSCode Server, shared memory)