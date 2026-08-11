set -x

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export VLLM_USE_V1=0
DATA_ROOT=/devsft_AFS/hanxiaoli/verl_data

JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3-8B-GRM}"
JUDGE_PORT="${JUDGE_PORT:-7001}"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://127.0.0.1:$JUDGE_PORT/v1}"
JUDGE_MAX_OUTPUT_LENGTH="${JUDGE_MAX_OUTPUT_LENGTH:-4096}"
JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-128}"

# Data preparation scripts are available in ``examples/data_preprocess``.
python3 examples/data_preprocess/deceptive_roles.py \
    --local_dir $DATA_ROOT/deceptive_roles_improved \
    --source_dir agent_system/environments/env_package/reason_chat/deceptive_roles_improved

train_files=$DATA_ROOT/deceptive_roles_improved/train.parquet
test_files=$DATA_ROOT/deceptive_roles_improved/test.parquet

CHECKPOINT_CONTENTS=['model','optimizer','extra'] # save hf_model

# Maximin RL training between agent and monitor
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
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    monitor_rollout_ref.enable=True \
    monitor_rollout_ref.enable_train_monitor=False \
    monitor_rollout_ref.model.path=google/gemma-3-4b-it \
    monitor_rollout_ref.model.attn_implementation=eager \
    monitor_rollout_ref.model.use_remove_padding=False \
    monitor_rollout_ref.model.use_fused_kernels=False \
    monitor_rollout_ref.model.freeze_vision_tower=True \
    monitor_rollout_ref.model.freeze_multi_modal_projector=True \
    monitor_rollout_ref.monitor.fsdp_config.use_orig_params=True \
    monitor_rollout_ref.monitor.ulysses_sequence_parallel_size=1 \
    monitor_rollout_ref.rollout.enforce_eager=True \
    monitor_rollout_ref.rollout.enable_chunked_prefill=False \
    monitor_rollout_ref.rollout.n=1 \
    monitor_rollout_ref.model.enable_gradient_checkpointing=True \
    monitor_rollout_ref.monitor.fsdp_config.param_offload=False \
    monitor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    monitor_rollout_ref.rollout.name=vllm \
    monitor_rollout_ref.rollout.gpu_memory_utilization=0.92 \
    monitor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    monitor_rollout_ref.rollout.val_kwargs.do_sample=True \
    monitor_rollout_ref.critique_link_check=False \
    reward_model.enable=True \
    reward_model.model.path=sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=32 \
    reward_model.reward_manager=episode \
    reward_model.normalization.enable=True \
    reward_model.normalization.rollout_overrides.temperature=1.1 \
    reward_model.normalization.rollout_overrides.top_p=1.0 \
    judge_model.backend=api_cot \
    judge_model.template_name=strict5_cot \
    judge_model.score_profiles.issue_action.valid_tokens='["0","1","2","3","4"]' \
    judge_model.score_profiles.issue_action.token_weights='[-1.0,-0.5,0.0,0.5,1.0]' \
    judge_model.api_cot.provider=openai_compatible \
    judge_model.api_cot.model=$JUDGE_MODEL_NAME \
    judge_model.api_cot.api_base=$JUDGE_API_BASE \
    judge_model.api_cot.api_key=dummy \
    judge_model.api_cot.temperature=0.6 \
    judge_model.api_cot.top_p=0.95 \
    judge_model.api_cot.top_k=20 \
    judge_model.api_cot.min_p=0.0 \
    judge_model.api_cot.presence_penalty=1.5 \
    judge_model.api_cot.max_output_length=$JUDGE_MAX_OUTPUT_LENGTH \
    judge_model.api_cot.max_concurrent=$JUDGE_MAX_CONCURRENT \
    judge_model.api_cot.timeout=180.0 \
    judge_model.api_cot.parse_error=neutral \
    judge_model.api_cot.api_error=raise \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=2e-6 \
    algorithm.lagrangian.enable=True \
    algorithm.lagrangian.lambda_init=1.0 \
    algorithm.lagrangian.lambda_max=4.0 \
    algorithm.lagrangian.lambda_lr=0.2 \
    algorithm.lagrangian.lambda_update_delay_steps=0 \
    algorithm.lagrangian.episode_cost_window_size=1024 \
    algorithm.lagrangian.threshold=0.3 \
    algorithm.lagrangian.adv_estimator=reinforce_plus_plus \
    monitor_rollout_ref.actor_cost_threshold=0.0 \
    env.env_name=ReasonChat \
    env.seed=0 \
    env.max_steps=1 \
    env.rollout.n=8 \
    env.rollout.val_n=1 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=4 \
    trainer.rollout_data_dir=auto \
    trainer.log_distributions=True \
    trainer.grm_judge_data_dir=null \
    trainer.grm_judge_data_freq=20 \
    trainer.project_name='verl_deceptive_roles' \
    trainer.experiment_name='grpo_gemma3_4b_monitor_cot_judge' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node_monitor=3 \
    trainer.nnodes_monitor=1 \
    trainer.save_steps='[140]' \
    trainer.test_freq=20 \
    trainer.total_epochs=60 \
    trainer.val_before_train=True $@
