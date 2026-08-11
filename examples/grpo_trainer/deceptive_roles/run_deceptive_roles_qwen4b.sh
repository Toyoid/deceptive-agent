set -x

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATA_ROOT=/devsft_AFS/hanxiaoli/verl_data

# Data preparation scripts are available in ``examples/data_preprocess``.
python3 examples/data_preprocess/deceptive_roles.py \
    --local_dir $DATA_ROOT/deceptive_roles_improved \
    --source_dir agent_system/environments/env_package/reason_chat/deceptive_roles_improved \
    --no_format_prompt

train_files=$DATA_ROOT/deceptive_roles_improved/train.parquet
test_files=$DATA_ROOT/deceptive_roles_improved/test.parquet

CHECKPOINT_CONTENTS=['model','optimizer','extra'] # save hf_model

# Vanilla RL training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=48 \
    data.val_batch_size=64 \
    data.max_prompt_length=256 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.actor.checkpoint.contents=$CHECKPOINT_CONTENTS \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward_model.enable=True \
    reward_model.model.path=sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=32 \
    reward_model.normalization.enable=True \
    reward_model.normalization.rollout_overrides.temperature=1.1 \
    reward_model.normalization.rollout_overrides.top_p=1.0 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=2e-5 \
    env.env_name=ReasonChat \
    env.seed=0 \
    env.max_steps=1 \
    env.rollout.n=8 \
    env.rollout.val_n=1 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=4 \
    trainer.rollout_data_dir=auto \
    trainer.project_name='verl_deceptive_roles' \
    trainer.experiment_name='grpo_qwen3_4b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_steps='[140]' \
    trainer.test_freq=20 \
    trainer.total_epochs=60 \
    trainer.val_before_train=True $@
