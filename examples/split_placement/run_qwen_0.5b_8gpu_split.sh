#!/bin/bash
set -x

# RLHF PPO training with Qwen 0.5B on 8 GPUs using split placement
# GPUs 0-5: Actor + Rollout (6 GPUs with tensor parallelism)
# GPUs 6-7: Critic + Reference Policy (2 GPUs)
# Optimized for high throughput training with better GPU utilization

# Activate conda environment with working dependencies
source /root/miniconda/etc/profile.d/conda.sh
conda activate verl_stable

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Increased for 8 GPUs
export OMP_NUM_THREADS=8  # Optimize for 8-GPU setup

PYTHONUNBUFFERED=1 python3 main_ppo_split.py \
    --config-path=/workspace/verl/examples/split_placement/config \
    --config-name=qwen_0.5b_8gpu_split \
    algorithm.adv_estimator=gae \
    data.train_files=/workspace/dataset/train_with_uid.parquet \
    data.val_files=/workspace/dataset/test_with_uid.parquet \
    data.train_batch_size=2048 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.universal_id_key=uid \
    data.reward_fn_key=data_source \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=6 \
    +actor_rollout_ref.actor.loss_agg_mode=token-mean \
    +actor_rollout_ref.actor.clip_ratio_low=0.2 \
    +actor_rollout_ref.actor.clip_ratio_high=0.2 \
    +actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra'] \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    +actor_rollout_ref.rollout.max_model_len=4096 \
    +actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    +actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    +actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    +actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    +actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    +actor_rollout_ref.rollout.val_kwargs.n=1 \
    +actor_rollout_ref.rollout.multi_turn.enable=false \
    +critic.rollout_n=1 \
    +critic.loss_agg_mode=token-mean \
    +critic.checkpoint.contents=['model','optimizer','extra'] \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.model.fsdp_config.fsdp_size=2 \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.ppo_max_token_len_per_gpu=32768 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_qwen_8gpu' \
    trainer.experiment_name='qwen_0.5b_8gpu_split_placement_v1' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.test_freq=10 \
    trainer.log_val_generations=10 \
    trainer.log_train_examples=3 \
    trainer.save_freq=5 $@ \
    2>&1 | tee verl_qwen0.5b_8gpu_split_placement_v1.log