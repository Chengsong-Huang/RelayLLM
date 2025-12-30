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

    # --- 阶段 1: 预处理并确定 Batch 状态 ---
    
    processed_data = []
    passed_prompts = set()
    
    # 标志 (Flags) 来跟踪 batch 的状态
    has_no_call_correct = False  # 是否存在 "不 call 且做对"
    has_call_correct = False     # 是否存在 "call 且做对"

    for reward_input in reward_inputs:
        prompt = reward_input["prompt"]
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        format_score = format_reward(response)
        
        is_correct = accuracy_score > 0.5 # 沿用 0.5 作为阈值
        
        # 沿用你代码中的 call 定义
        if_call = '<call>' in response
        call_num = sum(min(int(num), 4096) for num in re.findall(r"<call>\s*(\d+)\s*</call>", response))
        # 确保 response_length 存在且不为 0，避免除零错误
        response_length = reward_input["response_length"]
        if response_length == 0:
            response_length = 1
        call_ratio = call_num / response_length *0.1
        if call_ratio > 1:
            call_ratio = 0.1
            print(reward_input)
        # 存储预处理结果
        item_data = {
            "prompt": prompt,
            "accuracy_score": accuracy_score,
            "format_score": format_score,
            "if_call": if_call,
            "call_num": call_num,
            "call_ratio": call_ratio,
            "is_correct": is_correct
        }
        processed_data.append(item_data)

        # 更新 batch 状态标志
        if is_correct:
            passed_prompts.add(prompt) # 用于 pass@n 计算
            if if_call:
                has_call_correct = True
            else:
                has_no_call_correct = True
    
    # --- 阶段 2: 根据 Batch 状态应用条件逻辑打分 ---

    scores = []
    for item in processed_data:
        # 获取预处理的数据
        is_correct = item["is_correct"]
        if_call = item["if_call"]
        call_ratio = item["call_ratio"]
        
        overall_score = 0.0  # 默认分数
        if is_correct:
            overall_score = 1.0 - call_ratio
        else:
            overall_score = 0.0 + call_ratio

        pass_at_n_score = 1.0 if item["prompt"] in passed_prompts else 0.0

        scores.append(
            {
                "overall": overall_score,
                "format": item["format_score"],
                "mean@32": item["accuracy_score"], # 沿用你代码中的 "mean@32" 键
                "pass@32": pass_at_n_score,        # 沿用你代码中的 "pass@32" 键
                "call_ratio": item["call_ratio"],
                'if_call': item["if_call"],        # (bool)
                'call_num': item["call_num"],        # (int)
            }
        )

    return scores