#!/bin/bash

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RLHF PPO training with GSM8K dataset on 2 GPUs using split placement
# This script trains a Qwen2.5-7B-Instruct model on the GSM8K math dataset
# with rule-based reward function and UID tracking for enhanced monitoring

set -e

# Activate the verl_new conda environment (with Hydra 1.3.2)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl_new

# Set environment variables for better memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

# Change to the split placement directory
cd /root/verl/examples/split_placement

echo "Starting GSM8K PPO training with split placement..."
echo "Dataset: GSM8K with UID tracking"
echo "Model: Qwen2.5-7B-Instruct"
echo "GPUs: 2 (split placement)"
echo "Reward: Rule-based math solution verification"

# Run the GSM8K PPO training
PYTHONUNBUFFERED=1 python3 main_ppo_split.py \
    --config-name ppo_trainer_gsm8k_7b \
    data.train_files=/root/verl/dataset/train_with_uid.parquet \
    data.val_files=/root/verl/dataset/test_with_uid.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.universal_id_key=uid \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=3072 \
    actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=5e-7 \
    critic.model.path=Qwen/Qwen2.5-7B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_max_token_len_per_gpu=4096 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger='[console,wandb]' \
    trainer.project_name=verl_gsm8k \
    trainer.experiment_name=gsm8k_7b_2gpu_split \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    2>&1 | tee verl_gsm8k_7b_2gpu_training.log

echo "GSM8K PPO training completed!"
echo "Log file: verl_gsm8k_7b_2gpu_training.log"
echo "WandB project: verl_gsm8k"
echo "Experiment: gsm8k_7b_2gpu_split"