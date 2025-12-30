import json
import random
from transformers import AutoTokenizer
from huggingface_hub import login
from datasets import Dataset, DatasetDict

# Replace with your actual token
login(token='hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
json_file_path = 'sft_Qwen_Qwen3-1.7B.json' #example: 'sft_Qwen_Qwen3-1.7B.json'
username = 'your_username' #example: 'your_username'
# Ensure random and AutoTokenizer are imported in your main script if used elsewhere

def insert_special_string_between_tokens(
    tokenizer: AutoTokenizer,
    sentence: str,
    insertion_point: int | None = None  # Note: According to new requirements, this parameter is ignored inside the function
) -> str:
    """
    Inserts 0, 1, 2, or 3 special strings (e.g., "<call></call>") 
    between random tokens in the `sentence`.
    Returns the detokenized string.
    """

    # 1. Tokenize the original sentence
    original_ids = tokenizer.encode(sentence, add_special_tokens=False)
    
    # 2. Decide the number of insertions (0, 1, 2, or 3)
    n_insertions = random.randint(0, 3)

    # 3. If insertion count is 0, return the original sentence directly
    if n_insertions == 0:
        return sentence

    # 4. Select unique insertion points
    L = len(original_ids)
    
    # If original_ids has length L, there are L+1 possible insertion points (indices 0 to L)
    # (i.e., between L tokens, plus the very beginning and very end)
    # Ensure we do not sample more than L+1 points
    num_to_sample = min(n_insertions, L + 1)

    if num_to_sample == 0:
        # Theoretically only happens when n_insertions=0, but kept as a safety check
        return sentence

    # Get `num_to_sample` unique insertion points and sort them in descending order.
    # Sorting in descending order (e.g., [5, 2, 0]) ensures that inserting at index 5
    # does not affect the relative positions of subsequent indices 2 or 0.
    insertion_points = sorted(random.sample(range(L + 1), num_to_sample), reverse=True)

    # 5. Execute insertion
    modified_ids = list(original_ids)
    
    for point in insertion_points:
        # Generate a new random string for each insertion
        random_number = random.randint(1, 9)
        random_digit = random.randint(0, 3)
        
        # Note: The variables random_number/random_digit are defined but not currently used in the string below
        special_string = f"<call></call>"
        
        # Tokenize the special string
        special_ids = tokenizer.encode(special_string, add_special_tokens=False)
        
        # Insert the special_ids list at the selected point
        # Python list slice assignment [point:point] allows inserting a list at a specific index
        modified_ids[point:point] = special_ids

    # 6. Detokenize and return
    return tokenizer.decode(modified_ids)


# --- Main processing ---
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: The specified JSON file was not found.")
    exit()

try:
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B', trust_remote_code=True)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

print(f"First item in the data:\n{data[0]}\n")
new_data = []
for item in data:
    answer = item.get('answer', '')

    # Use the insertion_point computed from tokenized answer (middle)
    # (Note: This is passed to the function but ignored by the current logic)
    answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
    insertion_point = len(answer_token_ids) // 2

    modified_answer = insert_special_string_between_tokens(
        tokenizer,
        answer,
        insertion_point=insertion_point
    )

    new_item = {
        'question': item.get('question', '') + r"\nPlease reason step by step, and put your final answer within \boxed{}.",
        'answer': modified_answer,
        'system': ""
    }
    new_data.append(new_item)

train_dataset = Dataset.from_list(new_data)
dataset_dict = {"train": train_dataset}
config_name = f"sft_Qwen_Qwen3-1.7B"
dataset = DatasetDict(dataset_dict)
print(f"Pushing dataset to {username}/{config_name}...")
dataset.push_to_hub(f"{username}/{config_name}", private=True, config_name=config_name)
print("Push complete.")