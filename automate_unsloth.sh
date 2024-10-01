#!/bin/bash

# Define hyperparameter ranges
lora_r_values=(8)
lora_alpha_values=(16)
lora_dropout_values=(0.0)
lora_target_modules_values=("q_proj,k_proj,v_proj,o_proj")
use_4bit_quantization_values=(True)
max_seq_len_values=(256 512)
learning_rate_values=(1e-4)
lr_scheduler_type_values=("cosine")
warmup_ratio_values=(0.0)
max_grad_norm_values=(1.0)
per_device_train_batch_size_values=(1)
num_train_epochs=(1)

# Loop through combinations of values
for lora_r in "${lora_r_values[@]}"; do
  for lora_alpha in "${lora_alpha_values[@]}"; do
    for lora_dropout in "${lora_dropout_values[@]}"; do
      for lora_target_modules in "${lora_target_modules_values[@]}"; do
        for use_4bit_quantization in "${use_4bit_quantization_values[@]}"; do
          for max_seq_len in "${max_seq_len_values[@]}"; do
            for learning_rate in "${learning_rate_values[@]}"; do
              for lr_scheduler_type in "${lr_scheduler_type_values[@]}"; do
                for warmup_ratio in "${warmup_ratio_values[@]}"; do
                  for max_grad_norm in "${max_grad_norm_values[@]}"; do
                    for per_device_train_batch_size in "${per_device_train_batch_size_values[@]}"; do
                      for epoch in "${num_train_epochs[@]}"; do
                        # Call the training script with the current combination of parameters
                        bash run_unsloth.sh "$lora_r" "$lora_alpha" "$lora_dropout" "$lora_target_modules" "$use_4bit_quantization" \
                        "meta-llama/Meta-Llama-3.1-8B-Instruct" "$max_seq_len" "$learning_rate" "$lr_scheduler_type" "$warmup_ratio" \
                        "$max_grad_norm" "$per_device_train_batch_size" 1 "$num_train_epochs" "text" "llama-sft-lora-fsdp" True \
                        "HF_TOKEN_PLACEHOLDER" 1
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
