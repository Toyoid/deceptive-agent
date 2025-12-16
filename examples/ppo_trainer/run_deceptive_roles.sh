set -x

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="offline"
DATA_ROOT=/DATA/lhx

# Data preparation scripts are available in ``examples/data_preprocess``.
# Example usage:
#
python3 examples/data_preprocess/deceptive_roles.py
# python3 examples/data_preprocess/deceptive_roles.py --neutral_suffix

train_files=$DATA_ROOT/data/deceptive_roles/train.parquet
test_files=$DATA_ROOT/data/deceptive_roles/test.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=2 \
    data.val_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=Qwen/Qwen2.5-7B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    monitor_rollout_ref.enable=True \
    monitor_rollout_ref.enable_train_monitor=True \
    monitor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    monitor_rollout_ref.monitor.optim.lr=1e-6 \
    monitor_rollout_ref.model.use_remove_padding=True \
    monitor_rollout_ref.monitor.optim.lr_warmup_steps_ratio=0.1 \
    monitor_rollout_ref.monitor.ppo_mini_batch_size=2 \
    monitor_rollout_ref.monitor.ppo_micro_batch_size_per_gpu=1 \
    monitor_rollout_ref.monitor.use_kl_loss=True \
    monitor_rollout_ref.monitor.kl_loss_coef=0.01 \
    monitor_rollout_ref.model.enable_gradient_checkpointing=True \
    monitor_rollout_ref.monitor.fsdp_config.param_offload=False \
    monitor_rollout_ref.monitor.fsdp_config.optimizer_offload=False \
    monitor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    monitor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    monitor_rollout_ref.rollout.name=vllm \
    monitor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    monitor_rollout_ref.rollout.n=2 \
    monitor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    monitor_rollout_ref.rollout.val_kwargs.do_sample=False \
    monitor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    monitor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=True \
    reward_model.model.path=sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=1 \
    judge_model.model.path=Qwen/Qwen2.5-7B-Instruct \
    judge_model.model.use_remove_padding=True \
    judge_model.model.fsdp_config.param_offload=True \
    judge_model.micro_batch_size_per_gpu=1 \
    judge_model.valid_tokens='["0","1","2","3"]' \
    judge_model.token_weights='[0.0,0.33,0.66,1.0]' \
    judge_model.top_k=10 \
    judge_model.top_k_fallback_strategy='zero' \
    algorithm.use_kl_in_reward=False \
    env.env_name=ReasonChat \
    env.seed=0 \
    env.max_steps=1 \
    env.rollout.n=2 \
    env.rollout.val_n=1 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.resume_mode=disable \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_deceptive_roles' \
    trainer.experiment_name='ppo_qwen2.5_7b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node_monitor=3 \
    trainer.nnodes_monitor=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=7 \
    trainer.val_before_train=False $@