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

import base64
import json
import multiprocessing
import pickle
import zlib

# Reuse `run_test` for convenience
from verl.utils.reward_score.prime_code.testing_util import run_test


def _temp_run(in_outs, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(in_outs, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def check_correctness(in_outs, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(in_outs, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(in_outs["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list[0]


def compute_score(solution_str, ground_truth) -> float:
    """
    Compute score for LiveCodeBench problems.

    Args:
        solution_str: Generated solution containing Python code
        ground_truth: Dictionary containing 'ground_truth' key with compressed test cases, or string with test cases

    Returns:
        float: 1.0 if all test cases pass, 0.0 otherwise
    """
    # Extract Python code from solution
    solution = solution_str.split("```python")[-1].split("```")[0]

    # Handle ground_truth format - it can be a dict or string
    if isinstance(ground_truth, dict):
        if "ground_truth" in ground_truth:
            test_cases_data = ground_truth["ground_truth"]
        else:
            print(f"Error: Dictionary ground_truth missing 'ground_truth' key. Keys: {list(ground_truth.keys())}")
            return 0.0
    else:
        test_cases_data = ground_truth

    # extract test cases
    try:
        if isinstance(test_cases_data, str):
            # Try direct JSON parse first
            in_outs = json.loads(test_cases_data)
        else:
            print(f"Error: test_cases_data is not a string: {type(test_cases_data)}")
            return 0.0
    except Exception as e:
        # If JSON parsing fails, try decompression method
        try:
            in_outs = json.loads(pickle.loads(zlib.decompress(base64.b64decode(test_cases_data.encode("utf-8")))))
        except Exception as e2:
            print(f"Error loading test cases - JSON: {e}, Decompression: {e2}")
            return 0.0

    success = False
    try:
        res, metadata = check_correctness(in_outs=in_outs, generation=solution, timeout=6, debug=False)
        success = all(map(lambda x: x is True, res))
    except Exception:
        pass

    return 1.0 if success else 0.0
