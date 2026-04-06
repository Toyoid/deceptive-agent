set -ex

num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

DATA_ROOT="${DATA_ROOT:-/devsft_AFS/hanxiaoli/verl_data}"
SOURCE_ROOT="${SOURCE_ROOT:-$(pwd)/Targeted-Manipulation-and-Deception-in-LLMs}"
DATASET_NAME="${DATASET_NAME:-booking_assistance}"
DOMAINS="${DOMAINS:-flights,hotels,restaurants,events}"
BEHAVIOR_MODE="${BEHAVIOR_MODE:-honest}"
TEST_RATIO="${TEST_RATIO:-0.1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-120}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MONITOR_ROLLOUT_N="${MONITOR_ROLLOUT_N:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"

train_files="$DATA_ROOT/$DATASET_NAME/train.parquet"
test_files="$DATA_ROOT/$DATASET_NAME/test.parquet"

if [ -f "$train_files" ] && [ -f "$test_files" ]; then
    echo "Using existing parquet files: $train_files and $test_files"
else
    if [ ! -d "$SOURCE_ROOT" ]; then
        echo "Booking source root not found: $SOURCE_ROOT" >&2
        echo "Either upload train/test parquet to $DATA_ROOT/$DATASET_NAME or set SOURCE_ROOT correctly." >&2
        exit 1
    fi

    python3 examples/data_preprocess/booking_assistance.py \
        --local_dir "$DATA_ROOT/$DATASET_NAME" \
        --source_root "$SOURCE_ROOT" \
        --domains "$DOMAINS" \
        --behavior_mode "$BEHAVIOR_MODE" \
        --test_ratio "$TEST_RATIO"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    monitor_rollout_ref.enable=True \
    monitor_rollout_ref.enable_train_monitor=True \
    monitor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    monitor_rollout_ref.model.use_remove_padding=True \
    monitor_rollout_ref.monitor.checkpoint.contents='["model","optimizer","extra"]' \
    monitor_rollout_ref.monitor.optim.lr=1e-6 \
    monitor_rollout_ref.monitor.optim.lr_warmup_steps_ratio=0.1 \
    monitor_rollout_ref.monitor.use_kl_loss=True \
    monitor_rollout_ref.monitor.kl_loss_coef=0.01 \
    monitor_rollout_ref.model.enable_gradient_checkpointing=True \
    monitor_rollout_ref.monitor.fsdp_config.param_offload=False \
    monitor_rollout_ref.monitor.fsdp_config.optimizer_offload=False \
    monitor_rollout_ref.monitor.ppo_mini_batch_size=64 \
    monitor_rollout_ref.monitor.ppo_micro_batch_size_per_gpu=8 \
    monitor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    monitor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    monitor_rollout_ref.rollout.name=vllm \
    monitor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    monitor_rollout_ref.rollout.n=$MONITOR_ROLLOUT_N \
    monitor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    monitor_rollout_ref.rollout.val_kwargs.do_sample=False \
    monitor_rollout_ref.ref.fsdp_config.param_offload=True \
    monitor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward_model.enable=True \
    reward_model.model.path=sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=16 \
    reward_model.reward_manager=episode \
    reward_model.normalization.enable=True \
    reward_model.normalization.rollout_overrides.temperature=1.1 \
    reward_model.normalization.rollout_overrides.top_p=1.0 \
    judge_model.model.path=Qwen/Qwen2.5-7B-Instruct \
    judge_model.model.use_remove_padding=True \
    judge_model.model.fsdp_config.param_offload=True \
    judge_model.micro_batch_size_per_gpu=8 \
    judge_model.valid_tokens='["0","1","2","3","4"]' \
    judge_model.token_weights='[-1.0,-0.5,0.0,0.5,1.0]' \
    judge_model.top_k=2 \
    judge_model.template_name=strict5 \
    algorithm.use_kl_in_reward=False \
    algorithm.lagrangian.enable=True \
    algorithm.lagrangian.lambda_init=1.0 \
    algorithm.lagrangian.lambda_max=5.0 \
    algorithm.lagrangian.lambda_lr=0.1 \
    algorithm.lagrangian.lambda_update_delay_steps=40 \
    algorithm.lagrangian.episode_cost_window_size=1500 \
    algorithm.lagrangian.threshold=0.15 \
    algorithm.lagrangian.adv_estimator=reinforce_plus_plus_baseline \
    env.env_name=ReasonChat \
    env.seed=0 \
    env.max_steps=1 \
    env.rollout.n=$ROLLOUT_N \
    env.rollout.val_n=1 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.resume_mode=disable \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=6 \
    trainer.rollout_data_dir=auto \
    trainer.log_distributions=True \
    trainer.project_name='verl_booking_assistance' \
    trainer.experiment_name='grpo_qwen7b_booking_maximin_lag' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node_monitor=4 \
    trainer.nnodes_monitor=1 \
    trainer.judge_pool_mode='with_monitor' \
    trainer.n_gpus_per_node_judge=2 \
    trainer.nnodes_judge=1 \
    trainer.save_freq=500 \
    trainer.test_freq=20 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=True $@
