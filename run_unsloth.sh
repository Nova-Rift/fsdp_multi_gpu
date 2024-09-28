#!/bin/bash

# Install required packages
pip install torch==2.1.1 datasets==2.17.1 scipy==1.12.0 hf_transfer==0.1.5 huggingface-hub==0.25.0 wandb==0.16.3 wheel==0.44.0 transformers==4.44.2 accelerate==0.34.2 peft==0.12.0 "trl<0.9.0" bitsandbytes==0.43.3 deepspeed==0.15.1 einops==0.8.0 sentencepiece==0.2.0 nltk==3.9.1 xformers==0.0.23 unsloth==2024.8 flash-attn==2.6.3 --no-build-isolation

# Set environment variables
export HF_HOME='/workspace/persistent'
export HF_HUB_ENABLE_HF_TRANSFER='True'

# Set default values for arguments (in case not provided)
lora_r=${1:-8}
lora_alpha=${2:-16}
lora_dropout=${3:-0.0}
lora_target_modules=${4:-"q_proj,k_proj,v_proj,o_proj"}
use_4bit_quantization=${5:-True}
model_name_or_path=${6:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
max_seq_len=${7:-1024}
learning_rate=${8:-1e-3}
lr_scheduler_type=${9:-"cosine"}
warmup_ratio=${10:-0.0}
max_grad_norm=${11:-1.0}
per_device_train_batch_size=${12:-1}
gradient_accumulation_steps=${13:-1}
max_steps=${14:-20}
dataset_text_field=${15:-"text"}
output_dir=${16:-"llama-sft-lora-fsdp"}
use_peft_lora=${17:-True}
hf_token=${18:-"HF_TOKEN_PLACEHOLDER"}
num_process=${19:-1}

# Run the training script with passed or default arguments
python train_automated.py \
--lora_r "$lora_r" \
--lora_alpha "$lora_alpha" \
--lora_dropout "$lora_dropout" \
--lora_target_modules "$lora_target_modules" \
--use_4bit_quantization "$use_4bit_quantization" \
--model_name_or_path "$model_name_or_path" \
--max_seq_len "$max_seq_len" \
--learning_rate "$learning_rate" \
--lr_scheduler_type "$lr_scheduler_type" \
--warmup_ratio "$warmup_ratio" \
--max_grad_norm "$max_grad_norm" \
--per_device_train_batch_size "$per_device_train_batch_size" \
--gradient_accumulation_steps "$gradient_accumulation_steps" \
--max_steps "$max_steps" \
--dataset_text_field "$dataset_text_field" \
--output_dir "$output_dir" \
--use_peft_lora "$use_peft_lora" \
--hf_token "$hf_token" \
--num_process "$num_process" \
--seed 100 \
--seed 100 \
--seed 100 \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train" \
--logging_steps 1 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "no" \
--save_strategy "no" \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing False \
--per_device_eval_batch_size 1 \
--gradient_checkpointing True \
--use_reentrant False \
--use_flash_attn True \
--use_unsloth True
