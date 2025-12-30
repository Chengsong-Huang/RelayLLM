# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, List, Dict
from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    passed_prompts = set()
    for reward_input in reward_inputs:
        response_for_check = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        if accuracy_reward(response_for_check, reward_input["ground_truth"]) > 0.5:
            passed_prompts.add(reward_input["prompt"])

    scores = []
    for reward_input in reward_inputs:
        prompt = reward_input["prompt"]
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])

        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        overall_score = (1 - format_weight) * accuracy_score + format_weight * format_score
        
        pass_at_n_score = 1.0 if prompt in passed_prompts else 0.0
        if_call = '<call>' in response
        if_call_score = 1 if if_call else 0
        call_num = sum(int(num) for num in re.findall(r"<call>\s*(\d+)\s*</call>", response))
        call_ratio = call_num / reward_input["response_length"]
        count = {"acc&call": 0, "acc&no_call": 0, "no_acc&call": 0, "no_acc&no_call": 0}
        if accuracy_score == 1 and if_call == 0:
            overall_score = 1
            count["acc&no_call"] += 1
        elif accuracy_score == 1 and if_call == 1:
            overall_score = 1
            count["acc&call"] += 1
        elif accuracy_score == 0 and if_call == 1:
            overall_score = 0
            count["no_acc&call"] += 1
        elif accuracy_score == 0 and if_call == 0:
            overall_score = 0
            count["no_acc&no_call"] += 1
        # overall_score = overall_score * if_call_score
        call_ratio_score = call_ratio * if_call_score
        # count_score = sum(count.values()) / len(count)
        scores.append(
            {
                "overall": accuracy_score,
                "format": format_score,
                "mean@32": accuracy_score,
                "pass@32": pass_at_n_score,
                "call_ratio": call_ratio,
                'if_call': if_call,
                'call_num': call_num,
                # 'call_ratio_score': call_ratio_score,
                # 'count': count
            }
        )

    return scores