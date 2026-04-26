set -x

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE="offline"
DATA_ROOT=/devsft_AFS/hanxiaoli/verl_data

MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
PORT=7000
API_BASE=http://127.0.0.1:$PORT/v1
DATA_FILE=$DATA_ROOT/searchR1_processed_direct/test.parquet

# Start the local vLLM server in another terminal before running this eval:
# bash examples/api_rollout_eval/serve_local_vllm.sh $PORT 1 $MODEL_NAME
# API_BASE must match the serve port, and MODEL_NAME must match the served model name.

python3 -m agent_system.api_rollout_eval.main \
    env.env_name=deceptive_search \
    env.seed=0 \
    env.max_steps=4 \
    env.history_length=4 \
    env.deceptive_search.search_url=http://127.0.0.1:8000/retrieve \
    env.deceptive_search.audit_probability=1.0 \
    env.deceptive_search.reward_correct=1.0 \
    env.deceptive_search.reward_source=0.0 \
    env.deceptive_search.penalty_audit=0.0 \
    env.deceptive_search.penalty_disclosure=0.0 \
    env.deceptive_search.grounding_check=false \
    data.files=$DATA_FILE \
    data.batch_size=256 \
    model.provider=openai_compatible \
    model.model=$MODEL_NAME \
    model.api_base=$API_BASE \
    model.api_key=dummy \
    model.temperature=1.0 \
    model.top_p=1.0 \
    model.max_tokens=512 \
    model.max_concurrent=32 \
    model.timeout=120.0 \
    dump.output_dir=auto \
    dump.n_trajectories=16 \
    dump.policy=random \
    dump.include_steps=true \
    dump.log_generations=4 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=api_rollout_deceptive_search \
    trainer.experiment_name=qwen25_7b_local_vllm \
    "$@"
