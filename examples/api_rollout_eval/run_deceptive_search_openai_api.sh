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
DATA_FILE=$DATA_ROOT/searchR1_processed_direct/test.parquet

API_KEY_ARG=
if [ -n "$API_KEY" ]; then
    API_KEY_ARG=model.api_key=$API_KEY
fi

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
    data.batch_size=128 \
    model.provider=openai_compatible \
    model.model=$MODEL_NAME \
    model.api_base=$API_BASE \
    $API_KEY_ARG \
    model.api_key_env=$API_KEY_ENV \
    model.temperature=1.0 \
    model.top_p=1.0 \
    model.max_tokens=512 \
    model.max_concurrent=16 \
    model.timeout=120.0 \
    dump.output_dir=auto \
    dump.n_trajectories=128 \
    dump.policy=random \
    dump.include_steps=true \
    dump.log_generations=6 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=api_rollout_deceptive_search \
    trainer.experiment_name=gpt4o_openai_api \
    "$@"
