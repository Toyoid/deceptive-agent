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
DATA_FILE=$DATA_ROOT/deceptive_roles/train.parquet

# Start the local vLLM server in another terminal before running this eval:
# bash examples/api_rollout_eval/serve_local_vllm.sh $PORT 1 $MODEL_NAME
# API_BASE must match the serve port, and MODEL_NAME must match the served model name.

# Prepare the ReasonChat eval parquet if needed:
# python3 examples/data_preprocess/deceptive_roles.py --local_dir $DATA_ROOT/deceptive_roles

python3 -m agent_system.api_rollout_eval.main \
    env.env_name=ReasonChat \
    env.seed=0 \
    env.max_steps=1 \
    env.history_length=2 \
    data.files=$DATA_FILE \
    data.batch_size=188 \
    model.provider=openai_compatible \
    model.model=$MODEL_NAME \
    model.api_base=$API_BASE \
    model.api_key=dummy \
    model.temperature=1.0 \
    model.top_p=1.0 \
    model.max_tokens=2048 \
    model.max_concurrent=32 \
    model.timeout=120.0 \
    dump.output_dir=auto \
    dump.n_trajectories=188 \
    dump.policy=random \
    dump.include_steps=true \
    dump.log_generations=4 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=api_rollout_reasonchat \
    trainer.experiment_name=qwen25_7b_local_vllm \
    "$@"
