# python add_command_upload.py




python SFT_stage/train.py \
    --model_name "Qwen/Qwen3-0.6B" \
    --dataset_name "HINT-lab/sft_Qwen_Qwen3-0.6B" \
    --output_dir "./results_0.6B" \
    --max_seq_length 4096 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16

CUDA_VISIBLE_DEVICES=1 python utils/vllm_service.py --model_path Qwen/Qwen3-8B --port 7780 &

sleep 5m

MODEL_PATH=model_path #replace it with your local file path
echo ${STORAGE_PATH}
cd RL_stage
CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=8192 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=simple_06 \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/simple_06 \
    worker.rollout.max_num_batched_tokens=16384 \
    data.train_files=HINT-lab/8B_filtered_data \
    worker.reward.reward_function=examples/reward_function/math_help_group.py:compute_score \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    worker.rollout.port=7780

python RL_stage/script/model_merger.py --model_path model_path --output_path model_path/models/simple_06

bash eval/evaluate_forhelp.bash model_path 8B 7780 0