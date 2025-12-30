import os
import json
import re
import random
from typing import Any, List, Dict
from mathruler.grader import extract_boxed_content, grade_answer
from multiprocessing import Pool, set_start_method, Process
from transformers import AutoTokenizer
import numpy as np
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Your Reward Functions ---
# (Ensure mathruler is installed: pip install mathruler)

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    # As mentioned: "correctness passes the second check"
    # We assume grade_answer is that check
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

# --- Configuration ---
MODEL_PATH = "Qwen/Qwen3-8B"  # Replace with your model path
DATASET_NAME = "guanning-ai/dapo17k"
DATASET_SPLIT = "train"  # or "test"
OUTPUT_DIR = "inference_results"

# vLLM Sampling Parameters
# n=16 means generating 16 independent outputs for each prompt
SAMPLING_PARAMS = SamplingParams(
    n=16,
    temperature=0.7,
    top_p=0.95,
    max_tokens=8192,
    # use_beam_search=False, # Ensure beam search is off
)

# --- Worker Function (Updated) ---

def process_shard(args):
    """
    Worker function to process a data shard on a single GPU.
    """
    gpu_id, dataset_shard, model_path, output_dir, tokenizer = args
    
    # Critical: Isolate GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] Process started, will process {len(dataset_shard)} items.")

    # Initialize vLLM inside the process
    try:
        # tensor_parallel_size=1 ensures vLLM uses only the single GPU it can see
        llm = LLM(model=model_path, tensor_parallel_size=1)
    except Exception as e:
        print(f"[GPU {gpu_id}] vLLM initialization failed: {e}")
        return
        
    print(f"[GPU {gpu_id}] Model loaded.")

    # Prepare prompts and ground truths
    # (Note: Modify 'problem' and 'answer' based on actual dapo17k column names)
    
    # 1. Apply your specified chat template
    prompt_suffix = r"\nPlease reason step by step, and put your final answer within \\boxed{}."
    prompts_formatted = [
        [{"role": "user", "content": item['problem'] + prompt_suffix}]
        for item in dataset_shard
    ]
    prompts = tokenizer.apply_chat_template(
        prompts_formatted, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=False
    )
    
    # Extract original problems and answers for later use
    original_problems = [item['problem'] for item in dataset_shard]
    ground_truths = [item['answer'] for item in dataset_shard]

    # Batch generation
    # vLLM will automatically apply the tokenizer's chat template (if the model has one)
    # to handle this list of chat dicts
    outputs = llm.generate(prompts, SAMPLING_PARAMS)
    
    print(f"[GPU {gpu_id}] Generation complete, starting filtering...")

    # Process results and save
    output_file_path = os.path.join(output_dir, f"results_gpu_{gpu_id}.jsonl")
    
    with open(output_file_path, "w", encoding="utf-8") as f_out:
        for i, request_output in enumerate(tqdm(outputs, desc=f"GPU {gpu_id} Filtering")):
            original_problem = original_problems[i]
            ground_truth = ground_truths[i]
            
            num_correct = 0
            
            # request_output.outputs is a list containing 16 CompletionOutput objects
            for completion_output in request_output.outputs:
                response_text = completion_output.text
                
                # Filter using your accuracy_reward function
                if accuracy_reward(response_text, ground_truth) == 1.0:
                    num_correct += 1
            
            # 2. Calculate pass rate
            success_rate = num_correct / SAMPLING_PARAMS.n  # SAMPLING_PARAMS.n is 16
            if success_rate > 0.0:
                result_entry = {
                    "problem": original_problem,
                    "success_rate": success_rate,
                    "answer": ground_truth
                }
                f_out.write(json.dumps(result_entry) + "\n")

    print(f"[GPU {gpu_id}] Processing complete. Results saved to {output_file_path}")

# --- Main Function ---

def main():
    # Check available GPUs
    if not torch.cuda.is_available():
        print("Error: CUDA not detected. This script requires GPUs.")
        return
        
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} available GPUs.")

    # Load dataset
    print(f"Loading dataset {DATASET_NAME} (split: {DATASET_SPLIT})...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # For testing, you can take a small slice here
    print(f"Dataset loaded, total {len(dataset)} items.")

    # Split dataset into n_gpus shards
    dataset_shards = np.array_split(dataset, n_gpus)
    
    # Prepare output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare arguments for each worker process
    pool_args = []
    for gpu_id in range(n_gpus):
        shard = dataset_shards[gpu_id]
        if len(shard) > 0:
            pool_args.append((
                gpu_id, 
                list(shard),  # Convert Hugging Face dataset to list
                MODEL_PATH, 
                OUTPUT_DIR,
                tokenizer
            ))

    # Set multiprocessing start method to 'spawn'
    # Must be called before starting any process
    set_start_method("spawn", force=True)
    print(f"Starting {len(pool_args)} worker processes...")

    # --- Deprecated Pool, using manual Process instead ---
    
    processes: List[Process] = []
    
    # 1. Start all processes
    for args_tuple in pool_args:
        # Note: args must be (args_tuple,) to pass correctly
        p = Process(target=process_shard, args=(args_tuple,)) 
        
        # Critical fix: Set process as non-daemon
        p.daemon = False 
        
        p.start()
        processes.append(p)

    # 2. Wait for all processes to complete
    for p in processes:
        p.join() 

    print("All processes completed.")
    
    # (Optional) Merge all result files
    print("Merging result files...")
    all_results = []
    for gpu_id in range(n_gpus):
        shard_file = os.path.join(OUTPUT_DIR, f"results_gpu_{gpu_id}.jsonl")
        if os.path.exists(shard_file):
            with open(shard_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    all_results.append(json.loads(line))
    
    final_output_file = os.path.join(OUTPUT_DIR, "all_results.jsonl")
    with open(final_output_file, "w", encoding="utf-8") as f_final:
        for entry in all_results:
            f_final.write(json.dumps(entry) + "\n")
            
    print(f"All results merged into {final_output_file}")


if __name__ == "__main__":
    main()