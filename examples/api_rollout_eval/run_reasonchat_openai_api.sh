set -x

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
DATA_ROOT=/devsft_AFS/hanxiaoli/verl_data

MODEL_NAME=gpt-4o
API_BASE=https://api.openai.com/v1
API_KEY_ENV=OPENAI_API_KEY
# API_KEY=
DATA_FILE=$DATA_ROOT/deceptive_roles/train.parquet

# Prepare the ReasonChat eval parquet if needed:
# python3 examples/data_preprocess/deceptive_roles.py --local_dir $DATA_ROOT/deceptive_roles

API_KEY_ARG=
if [ -n "$API_KEY" ]; then
    API_KEY_ARG=model.api_key=$API_KEY
fi

python3 -m agent_system.api_rollout_eval.main \
    env.env_name=ReasonChat \
    env.seed=0 \
    env.max_steps=1 \
    data.files=$DATA_FILE \
    data.batch_size=188 \
    model.provider=openai_compatible \
    model.model=$MODEL_NAME \
    model.api_base=$API_BASE \
    $API_KEY_ARG \
    model.api_key_env=$API_KEY_ENV \
    model.temperature=1.0 \
    model.top_p=1.0 \
    model.max_tokens=2048 \
    model.max_concurrent=16 \
    model.timeout=120.0 \
    dump.output_dir=auto \
    dump.n_trajectories=188 \
    dump.policy=random \
    dump.include_steps=true \
    dump.log_generations=4 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=api_rollout_reasonchat \
    trainer.experiment_name=gpt4o_openai_api \
    "$@"
