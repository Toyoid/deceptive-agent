#!/bin/bash
set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: run_gemma3_4b.sh <nproc_per_node> [other_configs...]"
    exit 1
fi

nproc_per_node=$1

shift 1

model_id=google/gemma-3-4b-it
MAX_LENGTH=2048

# Prereprocess self-monitor dataset
export HF_ENDPOINT="https://hf-mirror.com"
DATA_ROOT=/devsft_AFS/hanxiaoli/verl_data

python3 examples/data_preprocess/self_monitor_sft.py \
    --local_dir $DATA_ROOT/self_monitor_sft \
    --tokenizer $model_id \
    --max_length $MAX_LENGTH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_ROOT/self_monitor_sft/train.parquet \
    data.val_files=$DATA_ROOT/self_monitor_sft/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=$MAX_LENGTH \
    data.truncation=error \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$model_id \
    model.attn_implementation=eager \
    model.enable_gradient_checkpointing=true \
    model.freeze_vision_tower=true \
    model.freeze_multi_modal_projector=true \
    model.strategy=fsdp \
    model.fsdp_config.use_orig_params=true \
    +model.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=Gemma3DecoderLayer \
    model.use_liger=false \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.03 \
    optim.weight_decay=0.0 \
    optim.lr_scheduler=constant \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false \
    trainer.project_name=self_monitor_sft \
    trainer.experiment_name=gemma3_4b_it \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 \
    trainer.default_hdfs_dir=null \
    "$@"
