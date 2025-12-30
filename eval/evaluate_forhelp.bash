#!/bin/bash
export VLLM_DISABLE_COMPILE_CACHE=1
model_name=$1
larger_model=$2
MODEL_NAMES=(
  $model_name
)
port=$3
gpu_queue=$4
TASKS=(
  "math"
  "gsm8k" 
  "minerva"
  "olympiad"
  "aime2024"
  "aime2025"
)

GPU_QUEUE=($gpu_queue)
echo "Available GPUs: ${GPU_QUEUE[@]}"

declare -A pids

start_job() {
  local gpu_id="$1"
  local model="$2"
  local task="$3"

  echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] Start task [${task}] with model [${model}] on GPU [${gpu_id}] ..."

  # --- MODIFIED LINE ---
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  python -m evaluation.generate_withhelp --small_model "${model}" --dataset "${task}" --larger_model "${larger_model}" --large_model_url "http://127.0.0.1:${port}/generate" &

  pids["${gpu_id}"]=$!
}

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "==> Processing model: ${MODEL_NAME}"
    TASK_INDEX=0
    NUM_TASKS=${#TASKS[@]}

    while :; do
        while [ ${#GPU_QUEUE[@]} -gt 0 ] && [ ${TASK_INDEX} -lt ${NUM_TASKS} ]; do
            gpu_id="${GPU_QUEUE[0]}"
            GPU_QUEUE=("${GPU_QUEUE[@]:1}")

            task="${TASKS[${TASK_INDEX}]}"
            ((TASK_INDEX++))

            start_job "$gpu_id" "$MODEL_NAME" "$task"
        done

        if [ ${TASK_INDEX} -ge ${NUM_TASKS} ] && [ ${#pids[@]} -eq 0 ]; then
            break
        fi

        for gpu_id in "${!pids[@]}"; do
            pid="${pids[$gpu_id]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "==> [$(date '+%Y-%m-%d %H:%M:%S')] GPU [${gpu_id}] job finished with PID [${pid}]."
                unset pids["$gpu_id"]
                GPU_QUEUE+=("$gpu_id")
            fi
        done

        sleep 1
    done
done

# --- MODIFIED LINES ---
python -m evaluation.results_recheck --model_name $model_name --larger_model "${larger_model}" &


echo "==> All tasks have finished!"