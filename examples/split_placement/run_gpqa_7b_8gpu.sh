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

# RLHF PPO training with GPQA dataset on 8 GPUs using split placement
# This script trains a Qwen2.5-7B-Instruct model on the GPQA (Graduate-level Google-Proof Q&A) dataset
# GPUs 0-5: Actor + Rollout + Reference (6 GPUs)
# GPUs 6-7: Critic (2 GPUs)
# 8-GPU Split Placement: Optimized for high throughput training

set -e

# Activate the verl_new conda environment (with Hydra 1.3.2)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl_new

# Set environment variables for better memory management
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN

# Change to the split placement directory
cd /root/verl/examples/split_placement

echo "Starting GPQA PPO training with 8-GPU split placement..."
echo "Dataset: GPQA Diamond with UID tracking"
echo "Model: Qwen2.5-7B-Instruct"
echo "GPUs: 8 (split placement - 6 for Actor/Rollout/Ref, 2 for Critic)"
echo "Reward: Rule-based multiple choice scoring"

# Run the GPQA PPO training
PYTHONUNBUFFERED=1 python3 main_ppo_split_8gpu.py \
    --config-name ppo_trainer_gpqa_7b \
    data.train_files=/root/verl/dataset/gpqa/train_with_uid.parquet \
    data.val_files=/root/verl/dataset/gpqa/test_with_uid.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.universal_id_key=uid \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=1536 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-6 \
    critic.model.path=Qwen/Qwen2.5-7B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger='[console,wandb]' \
    trainer.project_name=verl_gpqa \
    trainer.experiment_name=gpqa_7b_8gpu_split \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=10 \
    2>&1 | tee verl_gpqa_7b_8gpu_training.log

echo "GPQA PPO training completed!"
echo "Log file: verl_gpqa_7b_8gpu_training.log"
echo "WandB project: verl_gpqa"
echo "Experiment: gpqa_7b_8gpu_split"