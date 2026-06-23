set -x

ENGINE=${1:-vllm}

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
# export WANDB_MODE="offline"
DATA_ROOT=/devsft_AFS/hanxiaoli/verl_data

CHECKPOINT_CONTENTS=['model','optimizer','extra'] # save hf_model for later maximin/monitor-only training
# Auxiliary prompt-only safety data preparation.
python3 -m verl.trainer.auxiliary.preprocess \
    prompt_only \
    --dataset PKU-Alignment/PKU-SafeRLHF \
    --split train \
    --local_dir $DATA_ROOT/pku_safe_rlhf/prompt_only \
    --data_source pku_safe_rlhf

TRAIN_DATA="$DATA_ROOT/searchR1_processed_direct/train.parquet"
VAL_DATA="$DATA_ROOT/searchR1_processed_direct/test.parquet"
aux_train_files=$DATA_ROOT/pku_safe_rlhf/prompt_only/train.parquet

# Main-task RL training with appended prompt-only auxiliary safety RL.
# The auxiliary data mix ratio is set by rollout sample counts:
#   aux_ratio = (auxiliary.batch_size * auxiliary.rollout.n) /
#               (data.train_batch_size * env.rollout.n + auxiliary.batch_size * auxiliary.rollout.n)
# With the values below: aux_ratio = (48 * 8) / (48 * 8 + 48 * 8) = 50%.

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=800 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.actor.checkpoint.contents=$CHECKPOINT_CONTENTS \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward_model.model.fsdp_config.param_offload=True \
    auxiliary.enable=True \
    auxiliary.start_step=1 \
    auxiliary.batch_size=96 \
    auxiliary.data.train_files="$aux_train_files" \
    auxiliary.data.max_prompt_length=512 \
    auxiliary.data.max_response_length=2048 \
    auxiliary.data.filter_overlong_prompts=True \
    auxiliary.data.truncation='error' \
    auxiliary.data.return_raw_chat=True \
    auxiliary.rollout.n=8 \
    auxiliary.reward_model.use_main=False \
    auxiliary.reward_model.strip_thinking=True \
    auxiliary.reward_model.model.path=sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    auxiliary.reward_model.model.use_remove_padding=True \
    auxiliary.reward_model.micro_batch_size_per_gpu=16 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=deceptive_search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=5 \
    env.history_length=4 \
    env.deceptive_search.search_url='http://127.0.0.1:8000/retrieve' \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=4 \
    trainer.project_name='verl_deceptive_search' \
    trainer.experiment_name='grpo_qwen3_8b_aux_safety_fresh_init' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_steps='[154]' \
    trainer.test_freq=155 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@
