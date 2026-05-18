set -x

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export WANDB_MODE="offline"
DATA_ROOT=/devsft_AFS/hanxiaoli/verl_data

TRAIN_DATA="$DATA_ROOT/verl-agent/text/train.parquet"
VAL_DATA="$DATA_ROOT/verl-agent/text/test.parquet"

JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen/Qwen3-8B}"
JUDGE_PORT="${JUDGE_PORT:-7001}"
JUDGE_API_BASE="${JUDGE_API_BASE:-http://127.0.0.1:$JUDGE_PORT/v1}"
JUDGE_MAX_OUTPUT_LENGTH="${JUDGE_MAX_OUTPUT_LENGTH:-4096}"
JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-256}"

num_cpus_per_env_worker=0.1

train_data_size=8
val_data_size=1

# Start the CoT judge server in another terminal before running this script.
# Example:
#   CUDA_VISIBLE_DEVICES=6,7 bash examples/grpo_trainer/serve_cot_judge_vllm.sh $JUDGE_PORT 2 $JUDGE_MODEL_NAME
# If the judge runs on the same node, reduce CUDA_VISIBLE_DEVICES here and
# adjust trainer GPU counts accordingly.

python3 -m examples.data_preprocess.prepare \
    --local_dir $DATA_ROOT/verl-agent \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=800 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=checkpoints/verl_agent_webshop/grpo_qwen3_4b_cheatshop/global_step_200/actor/huggingface \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    monitor_rollout_ref.enable=True \
    monitor_rollout_ref.enable_train_monitor=True \
    monitor_rollout_ref.rollout.n=1 \
    monitor_rollout_ref.algorithm.adv_estimator=reinforce_plus_plus \
    monitor_rollout_ref.data.max_prompt_length=8192 \
    monitor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    monitor_rollout_ref.data.truncation='error' \
    monitor_rollout_ref.model.path=Qwen/Qwen3-4B \
    monitor_rollout_ref.model.use_remove_padding=True \
    monitor_rollout_ref.model.enable_gradient_checkpointing=True \
    monitor_rollout_ref.model.chat_template_kwargs.enable_thinking=False \
    monitor_rollout_ref.monitor.optim.lr=2e-6 \
    monitor_rollout_ref.monitor.use_kl_loss=True \
    monitor_rollout_ref.monitor.kl_loss_coef=0.01 \
    monitor_rollout_ref.monitor.kl_loss_type=low_var_kl \
    monitor_rollout_ref.monitor.fsdp_config.param_offload=True \
    monitor_rollout_ref.monitor.fsdp_config.optimizer_offload=True \
    monitor_rollout_ref.monitor.use_invalid_action_penalty=True \
    monitor_rollout_ref.monitor.invalid_action_penalty_coef=0.1 \
    monitor_rollout_ref.monitor.ppo_mini_batch_size=24 \
    monitor_rollout_ref.monitor.ppo_micro_batch_size_per_gpu=2 \
    monitor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    monitor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    monitor_rollout_ref.rollout.name=$ENGINE \
    monitor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    monitor_rollout_ref.rollout.enable_chunked_prefill=False \
    monitor_rollout_ref.rollout.enforce_eager=False \
    monitor_rollout_ref.rollout.free_cache_engine=False \
    monitor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    monitor_rollout_ref.rollout.val_kwargs.do_sample=False \
    monitor_rollout_ref.ref.fsdp_config.param_offload=True \
    monitor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward_model.reward_manager=actor_monitor \
    judge_model.backend=api_cot \
    judge_model.template_name=strict5_cot \
    judge_model.valid_tokens='["0","1","2","3","4"]' \
    judge_model.token_weights='[-1.0,-0.5,0.0,0.5,1.0]' \
    judge_model.api_cot.provider=openai_compatible \
    judge_model.api_cot.model=$JUDGE_MODEL_NAME \
    judge_model.api_cot.api_base=$JUDGE_API_BASE \
    judge_model.api_cot.api_key=dummy \
    judge_model.api_cot.temperature=0.6 \
    judge_model.api_cot.top_p=0.95 \
    judge_model.api_cot.top_k=20 \
    judge_model.api_cot.min_p=0.0 \
    judge_model.api_cot.max_output_length=$JUDGE_MAX_OUTPUT_LENGTH \
    judge_model.api_cot.max_concurrent=$JUDGE_MAX_CONCURRENT \
    judge_model.api_cot.timeout=180.0 \
    judge_model.api_cot.parse_error=neutral \
    judge_model.api_cot.api_error=raise \
    algorithm.use_kl_in_reward=False \
    env.env_name=CheatShop \
    env.seed=0 \
    env.max_steps=9 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.cheatshop.require_self_report=True \
    env.cheatshop.enable_log_edit=False \
    env.cheatshop.monitor_obs_level=action_plus_obs \
    env.cheatshop.self_report_cot_visibility=hidden \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=4 \
    trainer.log_distributions=True \
    trainer.project_name='cheatshop' \
    trainer.experiment_name='grpo_qwen3_cheatshop_mm_cot_judge' \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node_monitor=3 \
    trainer.nnodes_monitor=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=1000 \
    trainer.val_before_train=True $@
