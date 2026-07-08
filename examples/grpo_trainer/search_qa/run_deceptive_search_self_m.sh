set -x

ENGINE=${1:-vllm}

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# export WANDB_MODE="offline"
DATA_ROOT=/devsft_AFS/hanxiaoli/verl_data

TRAIN_DATA="$DATA_ROOT/searchR1_processed_direct/train.parquet"
VAL_DATA="$DATA_ROOT/searchR1_processed_direct/test.parquet"

# A self-monitor SFT checkpoint is required before RL.
# self_monitor_sft_ckpt=checkpoints/self_monitor_sft/qwen3_8b/global_step_954
self_monitor_sft_ckpt=checkpoints/self_monitor_sft/qwen3_8b_cheating_search_agent/global_step_954

# Self-monitor RL training (actor-only constrained RL, no external monitor model)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=96 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$self_monitor_sft_ckpt" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward_model.reward_manager=actor_monitor \
    self_monitor.enable=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=deceptive_search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=5 \
    env.history_length=4 \
    env.deceptive_search.search_url='http://127.0.0.1:8000/retrieve' \
    env.deceptive_search.reward_correct=1.0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=4 \
    trainer.log_distributions=True \
    trainer.project_name='verl_deceptive_search' \
    trainer.experiment_name='self_monitor_grpo_on_cheating_plus_sft' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_steps='[120,200]' \
    trainer.test_freq=121 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False "$@"
