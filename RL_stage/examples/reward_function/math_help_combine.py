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
from typing import Any
import random
from mathruler.grader import extract_boxed_content, grade_answer
import re


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    flag = random.randint(0, 1)
    for reward_input in reward_inputs:

        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        call_num = sum(min(int(num), 4096) for num in re.findall(r"<call>\s*(\d+)\s*</call>", response))
        call_ratio = min(call_num / reward_input["response_length"], 1.0)
        if_call = call_ratio > 0.0
        if_call_score = 1 if if_call else 0
        count = {"acc&call": 0, "acc&no_call": 0, "no_acc&call": 0, "no_acc&no_call": 0}
        if accuracy_score == 1 and if_call == 0:
            overall_score = 1
            count["acc&no_call"] += 1
        elif accuracy_score == 0 and if_call == 1:
            overall_score = 0
            count["no_acc&call"] += 1
        elif accuracy_score == 0 and if_call == 0:
            overall_score = -1
            count["no_acc&no_call"] += 1
        elif accuracy_score == 1 and if_call == 1:
            overall_score = 1 - call_ratio
            count["acc&call"] += 1
        scores.append(
            {
                "overall": overall_score if flag == 0 else accuracy_score,
                "is_call": if_call_score,
                "call_ratio": call_ratio,
                "accuracy": accuracy_score,
                "format": format_score,
                "call_num": call_num,
                "acc&call": count["acc&call"],
                "acc&no_call": count["acc&no_call"],
                "no_acc&call": count["no_acc&call"],
                "no_acc&no_call": count["no_acc&no_call"],
            }
        )

    return scores
