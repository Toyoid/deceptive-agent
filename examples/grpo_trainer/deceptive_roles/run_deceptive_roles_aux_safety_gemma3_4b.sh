set -x

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=0

DATA_ROOT=/devsft_AFS/hanxiaoli/verl_data

# Main-task data preparation.
python3 examples/data_preprocess/deceptive_roles.py \
    --local_dir $DATA_ROOT/deceptive_roles_improved \
    --source_dir agent_system/environments/env_package/reason_chat/deceptive_roles_improved \
    --suffix_prompt

# Auxiliary prompt-only safety data preparation.
python3 -m verl.trainer.auxiliary.preprocess \
    prompt_only \
    --dataset PKU-Alignment/PKU-SafeRLHF \
    --split train \
    --local_dir $DATA_ROOT/pku_safe_rlhf/prompt_only \
    --data_source pku_safe_rlhf

train_files=$DATA_ROOT/deceptive_roles_improved/train.parquet
test_files=$DATA_ROOT/deceptive_roles_improved/test.parquet
aux_train_files=$DATA_ROOT/pku_safe_rlhf/prompt_only/train.parquet

CHECKPOINT_CONTENTS=['model','optimizer','extra'] # save hf_model

# Main-task RL training with appended prompt-only auxiliary safety RL.
# The auxiliary data mix ratio is set by rollout sample counts:
#   aux_ratio = (auxiliary.batch_size * auxiliary.rollout.n) /
#               (data.train_batch_size * env.rollout.n + auxiliary.batch_size * auxiliary.rollout.n)
# With the values below: aux_ratio = (48 * 8) / (48 * 8 + 48 * 8) = 50%.

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=62 \
    data.val_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=google/gemma-3-4b-it \
    actor_rollout_ref.model.attn_implementation=eager \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.freeze_vision_tower=True \
    actor_rollout_ref.model.freeze_multi_modal_projector=True \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward_model.enable=True \
    reward_model.model.path=sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=16 \
    reward_model.normalization.enable=True \
    reward_model.normalization.rollout_overrides.temperature=1.1 \
    reward_model.normalization.rollout_overrides.top_p=1.0 \
    auxiliary.enable=True \
    auxiliary.start_step=1 \
    auxiliary.batch_size=64 \
    auxiliary.data.train_files="$aux_train_files" \
    auxiliary.data.max_prompt_length=512 \
    auxiliary.data.max_response_length=1024 \
    auxiliary.data.filter_overlong_prompts=True \
    auxiliary.data.truncation='error' \
    auxiliary.data.return_raw_chat=True \
    auxiliary.rollout.n=8 \
    auxiliary.reward_model.use_main=True \
    auxiliary.reward_model.strip_thinking=True \
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
    trainer.experiment_name='grpo_gemma3_4b_aux_safety' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_steps='[140]' \
    trainer.test_freq=20 \
    trainer.total_epochs=60 \
    trainer.val_before_train=True $@
