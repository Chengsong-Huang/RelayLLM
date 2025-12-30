import re
import math
from collections import defaultdict
from typing import Any, List, Dict
import random
# ----------------------------------------------------------------------
# Try to import the official grader. If it fails, use a mock for demonstration.
# In your production environment, ensure 'mathruler' is installed.
# ----------------------------------------------------------------------

from mathruler.grader import grade_answer
print("Successfully imported 'grade_answer' from mathruler.")

def grade_answer_combined(predicted_answer: str, ground_truth: str) -> float:
    """
    Combined grading logic:
    1. First, perform a standardized exact match.
    2. If it doesn't match, use mathruler's grade_answer for mathematical equivalence check.
    """
    # Step 1: Exact match (ignoring leading/trailing whitespace and case)
    if predicted_answer.strip().lower() == ground_truth.strip().lower():
        return 1.0
    
    # Step 2: If exact match fails, call the more complex math grading function
    if grade_answer(predicted_answer, ground_truth):
        return 1.0
        
    return 0.0
# ----------------------------------------------------
# 1. Core reward calculation functions (as provided by you)
# ----------------------------------------------------
def is_equal(a: Any, b: Any) -> bool:
    """Simple equality check."""
    return grade_answer_combined(str(a), str(b))==1.0   

def calculate_rarity_scores_with_isequal(input_list: List[str], num_selections: int = 8) -> List[float]:
    if not input_list:
        return []
        
    scores = []
    for item_to_score in input_list:
        # Manually count occurrences using the custom is_equal function
        count = sum(1 for x in input_list if is_equal(x, item_to_score))
        
        # Apply the desired scoring formula
        score = 1 / (count + 1)
        scores.append(score)
        
    return scores

def calculate_perfect_scores_with_target_direct(input_list: List[str], target_number: str, num_selections: int = 8) -> List[float]:
    """Calculates the 'Target Contribution Score' for each item."""
    target_count = input_list.count(target_number)
    total_positions = len(input_list)
    avg_scores = []
    if not input_list: return []

    for i in range(total_positions):
        selection_probability = num_selections / total_positions
        
        target_in_selected_prob = 0.0
        if input_list[i] == target_number:
            target_in_selected_prob = 1.0
        else:
            remaining_targets = target_count
            remaining_positions = total_positions - 1
            other_selections = num_selections - 1
            
            if other_selections > remaining_positions:
                 target_in_selected_prob = 1.0
            elif other_selections > remaining_positions - remaining_targets:
                target_in_selected_prob = 1.0
            else:
                try:
                    no_target_ways = math.comb(remaining_positions - remaining_targets, other_selections)
                    total_ways = math.comb(remaining_positions, other_selections)
                    if total_ways > 0:
                        target_in_selected_prob = 1 - (no_target_ways / total_ways)
                except ValueError:
                    target_in_selected_prob = 0.0

        avg_score = selection_probability * target_in_selected_prob
        avg_scores.append(avg_score)
        
    return avg_scores

# ----------------------------------------------------
# 2. Answer extraction and grading functions
# ----------------------------------------------------
def extract_answer_from_solution(solution: str) -> str:
    """Extracts the answer from the solution field using the provided logic."""
    if not solution: return ""
    boxed_start = solution.find('\\boxed{')
    if boxed_start == -1:
        lines = solution.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if '=' in line and not line.startswith('\\'):
                parts = line.split('=')
                if len(parts) > 1: return parts[-1].strip()
        return ""
    content_start = boxed_start + len('\\boxed{')
    brace_count = 1; pos = content_start
    while pos < len(solution) and brace_count > 0:
        if solution[pos] == '{': brace_count += 1
        elif solution[pos] == '}': brace_count -= 1
        pos += 1
    if brace_count == 0:
        boxed_content = solution[content_start:pos-1].strip()
        return ' '.join(boxed_content.split())
    else:
        end_pos = solution.find('}', content_start)
        if end_pos != -1: return solution[content_start:end_pos].strip()
        return ""



# ----------------------------------------------------
# 3. Main compute_score function
# ----------------------------------------------------
def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    The main reward function, now updated to use the combined grading logic.
    - Groups inputs by 'prompt', assuming 8 items per group.
    - Calculates uniqueness and target_contribution scores for each group.
    - Calculates a base accuracy for each response.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Input must be a list.")

    # Group inputs by prompt
    grouped_by_prompt = defaultdict(list)
    for i, reward_input in enumerate(reward_inputs):
        # Store the original index to return results in the correct order
        reward_input['original_index'] = i
        grouped_by_prompt[reward_input["prompt"]].append(reward_input)

    # Initialize the results list to ensure correct ordering
    scores = [{} for _ in reward_inputs]

    # Iterate over each group (batch)
    for prompt, batch_items in grouped_by_prompt.items():

        num_selections = len(batch_items)
        if num_selections != 8:
            print(f"Warning: Prompt '{prompt[:30]}...' has {num_selections} samples, not 8. Proceeding with actual count.")
        
        if num_selections == 0:
            continue

        # Extract predicted answers and the ground truth answer
        predicted_answers = [extract_answer_from_solution(item['response']) for item in batch_items]
        # Assume ground_truth is the same for all items with the same prompt
        ground_truth_answer = batch_items[0]['ground_truth']
        # print(f"Predicted answers: {predicted_answers}")
        # print(f"Ground truth answer: {ground_truth_answer}")
        # Call the new reward functions to calculate scores
        uniqueness_scores = calculate_rarity_scores_with_isequal(predicted_answers, num_selections//2)
        target_scores = calculate_perfect_scores_with_target_direct(predicted_answers, ground_truth_answer, num_selections//2)
        # print(f"Uniqueness scores: {uniqueness_scores}")
        # print(f"Target scores: {target_scores}")
        # print("--------------------------------")
        # exit()
        # Combine scores and place them back into the results list
        if random.random() < 0.4:
            for i, item in enumerate(batch_items):
                # Calculate base accuracy using the combined grading logic
                accuracy = grade_answer_combined(predicted_answers[i], ground_truth_answer)
                
                # Get the original index to maintain order
                original_index = item['original_index']
                # Store all calculated scores
                scores[original_index] = {
                    "uniqueness_score": uniqueness_scores[i],
                    "target_contribution_score": target_scores[i],
                    "accuracy": accuracy,
                    'overall': uniqueness_scores[i] + target_scores[i] if len(predicted_answers[i])>0 else 0
                }
        else:
            for i, item in enumerate(batch_items):
                # Calculate base accuracy using the combined grading logic
                accuracy = grade_answer_combined(predicted_answers[i], ground_truth_answer)
                
                # Get the original index to maintain order
                original_index = item['original_index']
                # Store all calculated scores
                scores[original_index] = {
                    "uniqueness_score": uniqueness_scores[i],
                    "target_contribution_score": target_scores[i],
                    "accuracy": accuracy,
                    'overall': accuracy
                }

    return scores