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
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    return data_source, np.mean(score_lst)


def log_test_individual_lengths(dataset, config, output_dir="outputs"):
    """Log individual test set sample lengths to CSV for tracking"""
    import os
    import ast
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for idx, row in dataset.iterrows():
        uid = row.get('uid', f'test_sample_{idx}')
        
        # Parse prompt
        prompt_text = ""
        prompt_length = 0
        if config.data.prompt_key in row and pd.notna(row[config.data.prompt_key]):
            prompt_field = row[config.data.prompt_key]
            try:
                if isinstance(prompt_field, str) and prompt_field.startswith("[{"):
                    prompt_data = ast.literal_eval(prompt_field)
                    if isinstance(prompt_data, list) and len(prompt_data) > 0:
                        prompt_text = prompt_data[0].get('content', '')
                        prompt_length = len(prompt_text.split())
                else:
                    prompt_text = str(prompt_field)
                    prompt_length = len(prompt_text.split())
            except (ValueError, SyntaxError):
                prompt_text = str(prompt_field)
                prompt_length = len(prompt_text.split())
        
        # Parse responses
        response_texts = []
        response_lengths = []
        if config.data.response_key in row and pd.notna(row[config.data.response_key]):
            responses = row[config.data.response_key]
            if isinstance(responses, list):
                for resp in responses:
                    resp_text = str(resp)
                    response_texts.append(resp_text)
                    response_lengths.append(len(resp_text.split()))
            else:
                resp_text = str(responses)
                response_texts.append(resp_text)
                response_lengths.append(len(resp_text.split()))
        
        # Get ground truth if available
        ground_truth = ""
        if config.data.reward_model_key in row and pd.notna(row[config.data.reward_model_key]):
            try:
                reward_data = row[config.data.reward_model_key]
                if isinstance(reward_data, dict):
                    ground_truth = reward_data.get('ground_truth', '')
                elif isinstance(reward_data, str):
                    reward_dict = ast.literal_eval(reward_data)
                    ground_truth = reward_dict.get('ground_truth', '')
            except:
                pass
        
        # Create record for each response
        for i, (resp_text, resp_length) in enumerate(zip(response_texts, response_lengths)):
            result = {
                'universal_id': uid,
                'step': 'test_eval',
                'sample_idx': idx,
                'response_idx': i,
                'data_source': row.get(config.data.data_source_key, 'unknown'),
                'prompt': prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text,
                'response': resp_text[:200] + "..." if len(resp_text) > 200 else resp_text,
                'prompt_length': prompt_length,
                'response_length': resp_length,
                'total_length': prompt_length + resp_length,
                'ground_truth': ground_truth,
                'sequence_score': 0.0,  # Will be filled by evaluation
                'sequence_reward': 0.0  # Will be filled by evaluation
            }
            results.append(result)
    
    # Save to CSV
    test_df = pd.DataFrame(results)
    csv_filename = os.path.join(output_dir, "test_set_individual_lengths.csv")
    test_df.to_csv(csv_filename, index=False)
    print(f"Test set individual lengths saved to: {csv_filename}")
    
    # Print summary
    print(f"Test Set Length Analysis:")
    print(f"  Total samples: {len(test_df)}")
    print(f"  Avg prompt length: {test_df['prompt_length'].mean():.1f} words")
    print(f"  Avg response length: {test_df['response_length'].mean():.1f} words")
    print(f"  Avg total length: {test_df['total_length'].mean():.1f} words")
    
    return test_df

@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get('use_shm', False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)
    
    # Log individual test set lengths
    output_dir = config.get('output_dir', 'outputs')
    test_lengths_df = log_test_individual_lengths(dataset, config, output_dir)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    compute_score = get_custom_reward_fn(config)

    # Create remote tasks
    remote_tasks = [process_item.remote(compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)]

    # Process results as they come in with individual score tracking
    individual_scores = []
    with tqdm(total=total) as pbar:
        task_idx = 0
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                individual_scores.append({
                    'sample_idx': task_idx,
                    'data_source': data_source,
                    'score': score
                })
                task_idx += 1
                pbar.update(1)

    # Update test lengths CSV with scores
    if individual_scores:
        scores_df = pd.DataFrame(individual_scores)
        # Merge scores back into test lengths
        test_lengths_updated = test_lengths_df.merge(
            scores_df, 
            left_on=['sample_idx', 'data_source'], 
            right_on=['sample_idx', 'data_source'], 
            how='left'
        )
        # Update sequence_score column
        test_lengths_updated['sequence_score'] = test_lengths_updated['score'].fillna(0.0)
        test_lengths_updated = test_lengths_updated.drop('score', axis=1)
        
        # Save updated file
        updated_csv = os.path.join(output_dir, "test_set_individual_lengths_with_scores.csv")
        test_lengths_updated.to_csv(updated_csv, index=False)
        print(f"Test set lengths with scores saved to: {updated_csv}")

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)

    print(metric_dict)


if __name__ == "__main__":
    main()
