#!/bin/bash

# Define hyperparameter ranges
lora_r_values=(4 8 16)  # Example values for lora_r
lora_alpha_values=(16 32)  # Example values for lora_alpha
lora_dropout_values=(0.0 0.1)  # Example values for lora_dropout
lora_target_modules_values=("q_proj,k_proj" "q_proj,k_proj,v_proj")  # Example values
use_4bit_quantization_values=(True False)  # Example for use_4bit_quantization
max_seq_len_values=(1024 2048)  # Example values for max_seq_len
learning_rate_values=(1e-4 1e-3)  # Example values for learning_rate
lr_scheduler_type_values=("linear" "cosine")  # Example values for lr_scheduler_type
warmup_ratio_values=(0.0 0.1)  # Example values for warmup_ratio
max_grad_norm_values=(1.0 2.0)  # Example values for max_grad_norm
per_device_train_batch_size_values=(1 4)  # Example values for batch size
max_steps_values=(100 200)  # Example values for max_steps

# Call the training script with the current set of hyperparameters
bash run_unsloth.sh 8 16 0.0 "q_proj,k_proj,v_proj,o_proj" False \
"meta-llama/Meta-Llama-3.1-8B-Instruct" 1024 1e-3 "cosine" 0.0 \
1.0 1 1 20 "text" "llama-sft-lora-fsdp" True "HF_TOKEN_PLACEHOLDER" 1

# # Loop through all combinations of hyperparameters
# for lora_r in "${lora_r_values[@]}"; do
#   for lora_alpha in "${lora_alpha_values[@]}"; do
#     for lora_dropout in "${lora_dropout_values[@]}"; do
#       for lora_target_modules in "${lora_target_modules_values[@]}"; do
#         for use_4bit_quantization in "${use_4bit_quantization_values[@]}"; do
#           for max_seq_len in "${max_seq_len_values[@]}"; do
#             for learning_rate in "${learning_rate_values[@]}"; do
#               for lr_scheduler_type in "${lr_scheduler_type_values[@]}"; do
#                 for warmup_ratio in "${warmup_ratio_values[@]}"; do
#                   for max_grad_norm in "${max_grad_norm_values[@]}"; do
#                     for per_device_train_batch_size in "${per_device_train_batch_size_values[@]}"; do
#                       for max_steps in "${max_steps_values[@]}"; do

                            # # Call the training script with the current set of hyperparameters
                            # bash run_unsloth.sh 8 16 0.0 "q_proj,k_proj,v_proj,o_proj" False \
                            # "meta-llama/Meta-Llama-3.1-8B-Instruct" 1024 1e-3 "cosine" 0.0 \
                            # 1.0 1 1 20 "text" "llama-sft-lora-fsdp" True "HF_TOKEN_PLACEHOLDER" 1

#                       done
#                     done
#                   done
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done