set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
export WANDB_MODE="offline"

MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
API_BASE=http://127.0.0.1:7000/v1
NUM_EPISODES=128

python3 -m agent_system.api_rollout_eval.main \
    env.env_name=CheatShop \
    env.seed=0 \
    env.max_steps=8 \
    data.num_episodes=$NUM_EPISODES \
    data.batch_size=32 \
    model.provider=openai_compatible \
    model.model=$MODEL_NAME \
    model.api_base=$API_BASE \
    model.api_key=dummy \
    model.temperature=1.0 \
    model.top_p=1.0 \
    model.max_tokens=896 \
    model.max_concurrent=32 \
    model.timeout=120.0 \
    env.cheatshop.require_self_report=false \
    env.cheatshop.enable_log_edit=false \
    env.cheatshop.monitor_obs_level=action_plus_obs \
    env.cheatshop.self_report_cot_visibility=hidden \
    dump.output_dir=auto \
    dump.n_trajectories=128 \
    dump.policy=random \
    dump.include_steps=true \
    dump.log_generations=6 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=api_rollout_cheatshop \
    trainer.experiment_name=qwen25_7b_local_vllm \
    "$@"
