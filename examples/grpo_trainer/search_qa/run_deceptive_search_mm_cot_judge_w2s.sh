set -x

ENGINE=${1:-vllm}

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="online"
export WANDB_ENTITY="toy-collaborate"
DATA_ROOT=/ssd/work/verl_data

TRAIN_DATA="$DATA_ROOT/searchR1_processed_direct/train.parquet"
VAL_DATA="$DATA_ROOT/searchR1_processed_direct/test.parquet"

JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-/ssd/work/models/Qwen3-8B}"
JUDGE_PORT="${JUDGE_PORT:-7001}"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://127.0.0.1:$JUDGE_PORT/v1}"
JUDGE_MAX_OUTPUT_LENGTH="${JUDGE_MAX_OUTPUT_LENGTH:-4096}"
JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-128}"

# Start the CoT judge server in another terminal before running this script.
# For a single 8-GPU node, use 3/3/2 split:
#   CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm.sh $JUDGE_PORT 2 $JUDGE_MODEL_NAME
#
# This training script exposes GPUs 0-5 to Ray for actor+monitor pools only.
# Override TRAIN_CUDA_VISIBLE_DEVICES and trainer GPU counts if the judge server
# runs on a separate node or separate machine.

CHECKPOINT_CONTENTS=['model','optimizer','extra','hf_model'] # save hf_model

# Maximin RL training between agent and monitor on deceptive_search, using API CoT judge.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=120 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=examples/grpo_trainer/search_qa/checkpoints/verl_deceptive_search/grpo_deceptive_search_qwen3_8b_fresh_penalty0/global_step_180/actor/huggingface \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=96 \
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
    monitor_rollout_ref.enable=True \
    monitor_rollout_ref.enable_train_monitor=True \
    monitor_rollout_ref.rollout.n=1 \
    monitor_rollout_ref.algorithm.adv_estimator=reinforce_plus_plus \
    monitor_rollout_ref.data.truncation='left' \
    monitor_rollout_ref.model.path=/ssd/work/models/Qwen3-0.6B \
    monitor_rollout_ref.model.use_remove_padding=True \
    monitor_rollout_ref.model.enable_gradient_checkpointing=True \
    monitor_rollout_ref.model.chat_template_kwargs.enable_thinking=False \
    monitor_rollout_ref.monitor.optim.lr=5e-7 \
    monitor_rollout_ref.monitor.optim.lr_warmup_steps_ratio=0.1 \
    monitor_rollout_ref.monitor.use_kl_loss=True \
    monitor_rollout_ref.monitor.kl_loss_coef=0.001 \
    monitor_rollout_ref.monitor.kl_loss_type=low_var_kl \
    monitor_rollout_ref.monitor.entropy_coeff=0 \
    monitor_rollout_ref.monitor.fsdp_config.param_offload=True \
    monitor_rollout_ref.monitor.fsdp_config.optimizer_offload=True \
    monitor_rollout_ref.monitor.use_invalid_action_penalty=True \
    monitor_rollout_ref.monitor.invalid_action_penalty_coef=0.01 \
    monitor_rollout_ref.monitor.ppo_mini_batch_size=64 \
    monitor_rollout_ref.monitor.ppo_micro_batch_size_per_gpu=2 \
    monitor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    monitor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    monitor_rollout_ref.rollout.name=$ENGINE \
    monitor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    monitor_rollout_ref.rollout.enable_chunked_prefill=False \
    monitor_rollout_ref.rollout.enforce_eager=False \
    monitor_rollout_ref.rollout.free_cache_engine=False \
    monitor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    monitor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    monitor_rollout_ref.rollout.val_kwargs.top_k=20 \
    monitor_rollout_ref.rollout.val_kwargs.do_sample=False \
    monitor_rollout_ref.ref.fsdp_config.param_offload=True \
    monitor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward_model.reward_manager=actor_monitor \
    judge_model.backend=api_cot \
    judge_model.template_name=strict5_cot \
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
    algorithm.gamma=0.95 \
    env.env_name=deceptive_search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=5 \
    env.history_length=4 \
    env.deceptive_search.search_url='http://127.0.0.1:8000/retrieve' \
    env.deceptive_search.reward_correct=1.0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=2 \
    trainer.log_distributions=True \
    trainer.project_name='verl_deceptive_search' \
    trainer.experiment_name='grpo_deceptive_search_mm_cot_jud_qwen06b_to_8b' \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node_monitor=1 \
    trainer.nnodes_monitor=1 \
    trainer.save_steps='[20,60,90,100,110,120]' \
    trainer.test_freq=121 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False "$@"
