#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_4_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/verl_data/self_monitor_sft/train.parquet \
    data.val_files=$HOME/verl_data/self_monitor_sft/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=4096 \
    data.truncation=error \
    data.train_batch_size=256 \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen3-4B \
    optim.lr=2e-5 \
    optim.warmup_steps_ratio=0.03 \
    optim.weight_decay=0.0 \
    optim.lr_scheduler=constant \
    trainer.default_local_dir=$save_path \
    trainer.project_name=self-monitor-sft \
    trainer.experiment_name=self-monitor-sft-qwen-3-4b-instruct-sp2 \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
