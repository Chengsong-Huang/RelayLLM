#!/bin/bash

#BSUB -m jiaxinh02
#BSUB -R "rusage[mem=100GB]"
#BSUB -gpu "num=4"
#BSUB -J qwen3_4b_math_grpo
#BSUB -o output.%J.log
#BSUB -e error.%J.log

set -x
# export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

MODEL_PATH=/storage1/jiaxinh/Active/chengsong/call_for_help/sft_data_create/results_0.6B/checkpoint-938  # replace it with your local file path
echo ${STORAGE_PATH}
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=8192 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=group_new \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/group_new \
    worker.rollout.max_num_batched_tokens=20000 \
    worker.reward.reward_function=examples/reward_function/math_help_group_new.py:compute_score \
    worker.rollout.port=7778 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    trainer.find_last_checkpoint=false \

