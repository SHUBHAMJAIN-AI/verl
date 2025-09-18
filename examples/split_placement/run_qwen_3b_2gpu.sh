#!/bin/bash
set -x

# RLHF PPO training with Qwen 3B on 2 GPUs using split placement
# GPU 0: Actor + Rollout
# GPU 1: Critic + Reference Policy

# Activate conda environment with working dependencies
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl_new

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

PYTHONUNBUFFERED=1 python3 main_ppo_split.py \
    --config-path=/root/verl/examples/split_placement/config \
    --config-name=qwen_0.5b_2gpu_split \
    algorithm.adv_estimator=gae \
    data.train_files=/root/verl/dataset/train_with_uid.parquet \
    data.val_files=/root/verl/dataset/test_with_uid.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.universal_id_key=uid \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.loss_agg_mode=token-mean \
    +actor_rollout_ref.actor.clip_ratio_low=0.2 \
    +actor_rollout_ref.actor.clip_ratio_high=0.2 \
    +actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra'] \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    +actor_rollout_ref.rollout.max_model_len=1024 \
    +actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    +actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    +actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    +actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    +actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    +actor_rollout_ref.rollout.val_kwargs.n=1 \
    +actor_rollout_ref.rollout.multi_turn.enable=false \
    +critic.rollout_n=1 \
    +critic.loss_agg_mode=token-mean \
    +critic.checkpoint.contents=['model','optimizer','extra'] \
    critic.model.path=Qwen/Qwen2.5-3B-Instruct \
    critic.optim.lr=5e-6 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_max_token_len_per_gpu=4096 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_qwen_2gpu' \
    trainer.experiment_name='qwen_3b_split_test' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=5 $@ \
    2>&1 | tee verl_qwen3b_split_placement_run1.log