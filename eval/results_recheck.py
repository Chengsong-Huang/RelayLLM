import json
from mathruler.grader import extract_boxed_content, grade_answer
import openai
import requests
from tqdm import tqdm
import random
import argparse
import os
from openai import OpenAI
import concurrent.futures

# --- Configuration ---
# Set the maximum number of concurrent API calls you wish to run
MAX_WORKERS = 10
# ---------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--larger_model", type=str, default='-')
parser.add_argument("--fix_number", type=int, default=None)
args = parser.parse_args()

STORAGE_PATH = os.getenv("STORAGE_PATH")
if not STORAGE_PATH:
    print("Warning: Environment variable STORAGE_PATH is not set.")

# It is recommended to use os.getenv("OPENAI_API_KEY") for security
client = OpenAI(api_key='YOUR_API_KEY_HERE') 


def process_example(answer: str, response: str) -> str:
    try:
        example = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a math answer checker."},
                {"role": "user", "content": f"Hi, there is an answer: {answer}\n\n, and the ground truth answer is: {response}\n\n, please check whether the answer is correct or not, and return the **only** Yes or No."}
            ],
            "temperature": 0.1
        }
        completion = client.chat.completions.create(**example)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing answer: {str(e)[:100]}...")
        return "No"

new_results = []

for model_name in [args.model_name]:
    datasets_to_check = [
        "math", "gsm8k", "amc", "minerva", 
        "olympiad", "aime2024", 
        "aime2025"
    ]
    
    for dataset in datasets_to_check:
        print(f"\n--- Processing {model_name} on {dataset} ---")
        
        # Attempt two possible paths
        if args.fix_number is not None:
            primary_path = f'{STORAGE_PATH}/evaluation/{model_name.replace("/","_")}_{args.larger_model}/results_{dataset}_{args.fix_number}.json'
        else:
            primary_path = f'{STORAGE_PATH}/evaluation/{model_name.replace("/","_")}_{args.larger_model}/results_{dataset}.json'
            
        if args.fix_number is not None:
            fallback_path = f'{STORAGE_PATH}/evaluation/{model_name.replace("/","_")}/results_{dataset}_{args.fix_number}.json'
        else:
            fallback_path = f'{STORAGE_PATH}/evaluation/{model_name.replace("/","_")}/results_{dataset}.json'
        
        results = None
        try:
            with open(primary_path, 'r') as f:
                results = json.load(f)
            print(f"Loaded results from {primary_path}")
        except FileNotFoundError:
            try:
                with open(fallback_path, 'r') as f:
                    results = json.load(f)
                print(f"Loaded results from {fallback_path}")
            except FileNotFoundError:
                print(f"Error: File not found in either path: {primary_path} or {fallback_path}")
                continue # Skip this dataset
            except json.JSONDecodeError:
                print(f"Error: JSON decode failed for {fallback_path}")
                continue
            except Exception as e:
                print(f"Error loading {fallback_path}: {e}")
                continue
        except json.JSONDecodeError:
            print(f"Error: JSON decode failed for {primary_path}")
            continue
        except Exception as e:
            print(f"Error loading {primary_path}: {e}")
            continue

        if not results:
            print(f"No results loaded for {dataset}, skipping.")
            continue

        items_to_recheck = []
        for i, result in enumerate(results):
            if result.get('score', 1.0) < 0.5:
                items_to_recheck.append((i, result))

        if not items_to_recheck:
            print(f"No items to re-check for {dataset}.")
        else:
            print(f"Found {len(items_to_recheck)} items to re-check for {dataset}. Starting ThreadPool...")
            
            futures_to_index = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for index, item in items_to_recheck:
                    future = executor.submit(process_example, item['answer'], item['response'])
                    futures_to_index[future] = index

                for future in tqdm(concurrent.futures.as_completed(futures_to_index), total=len(items_to_recheck), desc=f"Checking {dataset}"):
                    original_index = futures_to_index[future]
                    try:
                        gpt_check_result = future.result() # Get API call result ("Yes" / "No")
                        
                        if "yes" in gpt_check_result.lower():
                            results[original_index]['score'] = 1
                            # print(results[original_index])
                    except Exception as e:
                        print(f"Error retrieving result for index {original_index}: {e}")

        try:
            final_score = round(sum([result.get('score', 0) for result in results]) / len(results) * 100, 2) if results else 0
        except ZeroDivisionError:
            print(f"Error: 'results' list is empty for {dataset}. Setting score to 0.")
            final_score = 0
        
        print(f"Final score for {dataset}: {final_score}")

        result_entry = {
            'model': model_name,
            'dataset': dataset,
            'score': final_score,
            'larger_model': args.larger_model
        }
        new_results.append(result_entry)
        
        try:
            with open(f'scores_recheck.jsonl', 'a') as f:
                json.dump(result_entry, f)
                f.write('\n')
        except IOError as e:
            print(f"Error writing to scores_recheck.jsonl: {e}")

print("\n--- Re-checking complete ---")
print("Final aggregated results:")
print(json.dumps(new_results, indent=2))