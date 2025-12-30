import argparse
import json
import os
import re
import time
from typing import List, Dict, Any

import requests
import vllm
import evaluation.datasets_loader as datasets_loader
from transformers import AutoTokenizer

STORAGE_PATH = os.getenv("STORAGE_PATH", ".")
# Remove global MAX_TURNS, manage or pass it through args in the function
GLOBAL_MAX_TOKENS = 8192

def call_large_model_service_batch(payloads: List[Dict], request_ids: List[int], url: str) -> List[Dict]:
    print(f"  [LM Batch Call IDs: {request_ids}] --> Delegating batch of {len(payloads)} requests...")

    def _create_error_response(msg: str) -> List[Dict]:
        return [{"text": f"[{msg}]", "finish_reason": "error"}] * len(payloads)

    try:
        response = requests.post(url, json=payloads, timeout=6000)
        
        if response.status_code == 200:
            results_data = response.json()

            if 'error' in results_data:
                error_msg = results_data.get('error', 'Unknown server error')
                print(f"  [LM Batch Call IDs: {request_ids}] <-- Error: Server returned an error: {error_msg}")
                return _create_error_response(f"Error: Server-side: {error_msg}")

            generation_results = results_data.get('results', [])
            print(f"  [LM Batch Call IDs: {request_ids}] <-- Large model responded successfully.")

            if len(generation_results) != len(payloads):
                return _create_error_response(f"Error: Mismatch count. Exp {len(payloads)}, got {len(generation_results)}")
            
            return generation_results
        else:
            return _create_error_response(f"Error: Service status {response.status_code}")
            
    except Exception as e:
        print(f"  [LM Batch Call IDs: {request_ids}] <-- Connection Error: {e}")
        return _create_error_response(f"Error: Connection failed: {e}")


def run_collaborative_generation(llm: vllm.LLM, tokenizer: AutoTokenizer, prompts: List[str], large_model_url: str) -> List[Dict]:
    MAX_TURNS = 5
    BASE_SMALL_MODEL_MAX_TOKENS = GLOBAL_MAX_TOKENS

    call_tag_pattern = re.compile(r"<call>\s*\d+\s*</call>")

    request_pool: List[Dict] = [{
        "id": i, 
        "base_prompt": prompt, 
        "current_solution": "",
        "status": "waiting_for_small_model", 
        "turn_count": 0,
        "current_token_count": 0,
        "delegate_info": None, 
        "final_answer": None,
        "large_model_contributions": []
    } for i, prompt in enumerate(prompts)]

    active_requests = len(request_pool)
    
    while active_requests > 0:
        print(f"\n{'='*20} New Loop Iteration | Active Requests: {active_requests} {'='*20}")

        # --- Small Model Step ---
        small_model_batch = [req for req in request_pool if req["status"] == "waiting_for_small_model"]
        if small_model_batch:
            print(f"Found {len(small_model_batch)} requests for small model.")
            
            current_prompts = []
            small_model_params_list = []
            batch_to_run = [] 

            for req in small_model_batch:
                remaining_tokens = GLOBAL_MAX_TOKENS - req['current_token_count']

                if remaining_tokens <= 0:
                    req['status'] = 'done'
                    req['final_answer'] = req['current_solution'] + f"\n[Error: Reached GLOBAL_MAX_TOKENS ({GLOBAL_MAX_TOKENS})]"
                    continue

                current_prompts.append(req['base_prompt'] + req['current_solution'])
                
                turn_max_tokens = min(BASE_SMALL_MODEL_MAX_TOKENS, remaining_tokens)
                
                small_model_params_list.append(
                    vllm.SamplingParams(
                        temperature=0.7, 
                        max_tokens=turn_max_tokens,
                        stop=["</call>"], 
                        include_stop_str_in_output=True
                    )
                )
                batch_to_run.append(req)

            if batch_to_run:
                outputs = llm.generate(current_prompts, small_model_params_list)
                
                for req, output in zip(batch_to_run, outputs):
                    req['turn_count'] += 1
                    small_model_output = output.outputs[0].text
                    
                    generated_token_count = len(tokenizer.encode(small_model_output, add_special_tokens=False))
                    req['current_token_count'] += generated_token_count
                    
                    finish_reason = output.outputs[0].finish_reason

                    has_call_tag = "<call>" in small_model_output

                    if (finish_reason == "stop" and not has_call_tag) or req['turn_count'] >= MAX_TURNS:
                        # Normal termination or maximum number of turns reached
                        req['status'] = 'done'
                        req['final_answer'] = req['current_solution'] + small_model_output
                        req['current_solution'] += small_model_output # 保持一致性
                        print(f"  [Req {req['id']}] --> FINISHED (Reason: {finish_reason}, Turn: {req['turn_count']}).")
                        continue

                    if has_call_tag:
                        req['current_solution'] += small_model_output
                        # Extract token count
                        match = re.search(r"<call>\s*(\d+)", small_model_output)
                        if match:
                            requested_lm_tokens = int(match.group(1))
                            # requested_lm_tokens = 20
                            remaining_tokens_for_lm = GLOBAL_MAX_TOKENS - req['current_token_count']
                            
                            if remaining_tokens_for_lm <= 0:
                                req['status'] = 'done'
                                req['final_answer'] = req['current_solution'] + f"\n[Error: Limit Reached]"
                            elif requested_lm_tokens == 0:
                                req['status'] = 'waiting_for_small_model' # 0 token 视为不调用
                            else:
                                capped_lm_tokens = min(requested_lm_tokens, remaining_tokens_for_lm)
                                req['status'] = 'waiting_for_large_model'
                                req['delegate_info'] = {"max_tokens": capped_lm_tokens}
                                print(f"  [Req {req['id']}] --> DELEGATING (Tokens: {capped_lm_tokens}).")
                        else:
                            req['status'] = 'failed'
                            req['final_answer'] = req['current_solution'] + "\n[Error: Bad <call> tag]"
                    else:
                        # finish_reason == "length" and no tag
                        req['current_solution'] += small_model_output
                        if finish_reason == "length":
                            print(f"  [Req {req['id']}] --> INTERRUPTED (Length). Continuing...")
                        else:
                            req['status'] = 'done'
                            req['final_answer'] = req['current_solution']

        # --- Large Model Step ---
        large_model_batch = [req for req in request_pool if req["status"] == "waiting_for_large_model"]
        if large_model_batch:
            payloads = []
            for req in large_model_batch:
                prompt_body_for_lm = call_tag_pattern.sub(" ", req['current_solution']).strip()
                
                payloads.append({
                    "prompt": req['base_prompt'] + prompt_body_for_lm, 
                    "max_tokens": req['delegate_info']['max_tokens'],
                })

            request_ids = [req['id'] for req in large_model_batch]
            large_model_results = call_large_model_service_batch(payloads, request_ids, large_model_url)
            
            for req, result_obj in zip(large_model_batch, large_model_results):
                result_text = result_obj.get("text", "")
                finish_reason = result_obj.get("finish_reason", "error")

                if finish_reason == "error" or result_text.startswith("[Error:"):
                    req['status'] = 'failed'
                    req['final_answer'] = req['current_solution'] + "\n" + result_text
                else:
                    contribution_len = len(tokenizer.encode(result_text, add_special_tokens=False))
                    prompt_body_for_lm = call_tag_pattern.sub(" ", req['current_solution']).strip()
                    start_token = len(tokenizer.encode(prompt_body_for_lm, add_special_tokens=False))
                    end_token = start_token + contribution_len
                    
                    req['large_model_contributions'].append({
                        "start_token": start_token,
                        "end_token": end_token
                    })
                    
                    req['current_token_count'] += contribution_len
                    req['current_solution'] += result_text 

                    if finish_reason == "stop":
                        req['status'] = 'done'
                        req['final_answer'] = req['current_solution']
                        print(f"  [LM ID: {req['id']}] <-- Large model FINISHED.")
                    elif finish_reason == "length":
                        if req['current_token_count'] >= GLOBAL_MAX_TOKENS:
                            req['status'] = 'done'
                            req['final_answer'] = req['current_solution'] + f"\n[Info: Limit Reached]"
                        else:
                            req['status'] = 'waiting_for_small_model'
                            print(f"  [LM ID: {req['id']}] <-- Large model LENGTH limit. Back to small.")
                    else:
                        req['status'] = 'waiting_for_small_model'

        active_requests = sum(1 for req in request_pool if req["status"] not in ["done", "failed"])

    request_pool.sort(key=lambda r: r['id'])
    return request_pool


def main(args):
    print(f"STORAGE_PATH: {STORAGE_PATH}")
    print(f"Small model: {args.small_model}, Dataset: {args.dataset}")
    
    print(f"Loading tokenizer from {args.small_model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.small_model, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Fallback to Qwen/Qwen3-0.6B only if intended.")
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)

    model = vllm.LLM(
        model=args.small_model,
        tokenizer=args.small_model,
        gpu_memory_utilization=0.85,
        trust_remote_code=True
    )
    
    handler = datasets_loader.get_dataset_handler(args.dataset, args.name)
    questions, answers = handler.load_data()

    prompts = []
    for question in questions:
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": question + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
            # enable_thinking=False 只对特定模型有效，加个 try-except 或移除
            try:
                p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                 p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(p)
        else:
            # Fallback
            prompts.append("User: " + question + "\nPlease reason step by step, and put your final answer within \\boxed{}.\nAssistant:")

    time_start = time.time()
    print("\nStarting collaborative generation process...")
    request_pool = run_collaborative_generation(model, tokenizer, prompts, args.large_model_url)
    print("Collaborative generation finished.\n")
    time_end = time.time()

    responses = [r['final_answer'] if r['final_answer'] is not None else r['current_solution'] for r in request_pool]
    all_contributions = [r['large_model_contributions'] for r in request_pool]

    scores, average_score = handler.get_score(responses, answers)

    results = []
    larger_model_tokens = 0
    total_tokens = 0

    for i, (q, a, r, s, contribs) in enumerate(zip(questions, answers, responses, scores, all_contributions)):
        results.append({
            "question": q, 
            "answer": a, 
            "response": r, 
            "score": s, 
            "large_model_contributions": contribs
        })
        
        if contribs:
            for c in contribs:
                larger_model_tokens += (c['end_token'] - c['start_token'])
        
        if r:
            total_tokens += len(tokenizer.encode(r, add_special_tokens=False))

    print(f"Average score: {average_score}")

    with open('final_results.jsonl', 'a') as f:
        json.dump({
            'model': args.small_model,
            'dataset': args.dataset,
            'score': round(average_score*100, 2),
            'larger_model': args.larger_model,
            'real_call_tokens': larger_model_tokens,
            'total_tokens': total_tokens,
            'real_call_tokens_ratio': (larger_model_tokens / total_tokens) if total_tokens > 0 else 0,
            'time': time_end - time_start
        }, f)
        f.write('\n')

    output_dir = f"{STORAGE_PATH}/evaluation/{args.small_model.replace('/', '_')}_{args.larger_model}"
    os.makedirs(output_dir, exist_ok=True)

    final_output_path = f"{output_dir}/results_{args.dataset}.json"
    
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {final_output_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--small_model", type=str, default="../LLaMA-Factory/saves/qwen3-0.6b-base/full/sft/checkpoint-500")
    parser.add_argument("--large_model_url", type=str, default="http://127.0.0.1:7778/generate")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--larger_model", type=str, default='7B')
    args = parser.parse_args()
    output_dir = f"{STORAGE_PATH}/evaluation/{args.small_model.replace('/', '_')}_{args.larger_model}"
    check_path = f"{output_dir}/results_{args.dataset}.json"
    main(args)