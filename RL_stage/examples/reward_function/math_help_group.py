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
        call_num = sum(min(500, 4096) for num in re.findall(r"<call>\s*(\d+)\s*</call>", response))
        call_ratio = min(call_num / reward_input["response_length"], 1.0)
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

        # --- 应用你的条件逻辑 ---

        if has_no_call_correct:
            # Case 1: "这个batch里面如果有不call也做对的"
            if is_correct and not if_call:
                overall_score = 1.5               # "不call做对的给1.5"
            elif is_correct and if_call:
                overall_score = 1.0 - call_ratio  # "call做对的给1-call_ratio"
            else:
                overall_score = 0.0               # "剩下的都是0" (做错的)

        elif has_call_correct:
            # Case 2: "如果只有call了做对的没有不call做对的"
            if is_correct and if_call:
                overall_score = 1.0 - call_ratio  # "做对的给1-call_ratio"
            elif not is_correct and not if_call:
                overall_score = -1.0              # "没call还错的给-1"
            elif not is_correct and if_call:
                overall_score = 0.0               # "做错的0" (这里指 call了但做错)

        else:
            # Case 3: "如果整个batch都没有做对的" (has_no_call_correct 和 has_call_correct 均为 False)
            if if_call:
                overall_score = call_ratio        # "给call的一个call_ratio的奖励"
            else:
                overall_score = 0.0               # (没 call 也没做对，默认为 0)

        # --- 组装最终得分字典 ---
        
        # 计算 pass@n
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