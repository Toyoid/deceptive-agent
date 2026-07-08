set -x

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
export WANDB_MODE="online"
export WANDB_ENTITY="toy-collaborate"
DATA_ROOT=/ssd/work/verl_data

python3 -m examples.data_preprocess.prepare \
    --local_dir $DATA_ROOT/verl-agent \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# Auxiliary prompt-only safety data preparation.
python3 -m verl.trainer.auxiliary.preprocess \
    prompt_only \
    --dataset PKU-Alignment/PKU-SafeRLHF \
    --split train \
    --local_dir $DATA_ROOT/pku_safe_rlhf/prompt_only \
    --data_source pku_safe_rlhf


TRAIN_DATA="$DATA_ROOT/verl-agent/text/train.parquet"
VAL_DATA="$DATA_ROOT/verl-agent/text/test.parquet"
aux_train_files=$DATA_ROOT/pku_safe_rlhf/prompt_only/train.parquet

num_cpus_per_env_worker=0.1

train_data_size=16
val_data_size=128

CHECKPOINT_CONTENTS=['model','optimizer','extra','hf_model'] # save hf_model for later maximin/monitor-only training

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=5120 \
    data.max_response_length=800 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/ssd/work/models/Qwen3-8B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.actor.checkpoint.contents=$CHECKPOINT_CONTENTS \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8960 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward_model.model.fsdp_config.param_offload=True \
    auxiliary.enable=True \
    auxiliary.start_step=1 \
    auxiliary.batch_size=16 \
    auxiliary.data.train_files="$aux_train_files" \
    auxiliary.data.max_prompt_length=512 \
    auxiliary.data.max_response_length=2048 \
    auxiliary.data.filter_overlong_prompts=True \
    auxiliary.data.truncation='error' \
    auxiliary.data.return_raw_chat=True \
    auxiliary.rollout.n=8 \
    auxiliary.reward_model.use_main=False \
    auxiliary.reward_model.strip_thinking=True \
    auxiliary.reward_model.model.path=/ssd/work/models/FsfairX-LLaMA3-RM-v0.1 \
    auxiliary.reward_model.model.use_remove_padding=True \
    auxiliary.reward_model.micro_batch_size_per_gpu=16 \
    algorithm.use_kl_in_reward=False \
    env.env_name=CheatShop \
    env.seed=0 \
    env.max_steps=12 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.cheatshop.require_self_report=False \
    env.cheatshop.enable_log_edit=False \
    env.cheatshop.self_report_cot_visibility=hidden \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.log_val_generations=2 \
    trainer.log_distributions=True \
    trainer.project_name='cheatshop' \
    trainer.experiment_name='grpo_qwen8b_cheatshop_aux_safety' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_steps='[40,80,100]' \
    trainer.test_freq=20 \
    trainer.total_epochs=200 \
    trainer.val_before_train=True $@
