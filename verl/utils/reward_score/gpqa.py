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

# Extraction Template from https://github.com/openai/simple-evals/blob/90e3e821cabba2aeb6be651dcb662b253df04225/common.py#L25
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"


def compute_score(solution_str, ground_truth) -> float:
    """
    Compute the score for GPQA multiple choice questions.

    Args:
        solution_str: The model's response string
        ground_truth: The correct answer (A, B, C, or D) or dictionary containing ground_truth key

    Returns:
        float: 1.0 if the extracted answer matches ground truth, 0.0 otherwise
    """
    # Handle ground_truth format - it can be a dict or string
    if isinstance(ground_truth, dict):
        if "ground_truth" in ground_truth:
            target_answer = ground_truth["ground_truth"]
        else:
            print(f"Error: Dictionary ground_truth missing 'ground_truth' key. Keys: {list(ground_truth.keys())}")
            return 0.0
    else:
        target_answer = ground_truth

    match = re.search(ANSWER_PATTERN_MULTICHOICE, solution_str)
    extracted_answer = match.group(1) if match else None
    score = 1.0 if extracted_answer == target_answer else 0.0
    return score
