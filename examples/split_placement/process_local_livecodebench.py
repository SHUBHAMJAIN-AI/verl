#!/usr/bin/env python3
"""
Process local LiveCodeBench jsonl file for verl training with UID tracking.
"""

import argparse
import base64
import json
import os
import pickle
import random
import uuid
import zlib

import pandas as pd


def add_uid_to_example(example, idx, data_source, ability, split):
    """
    Add UID to each example for tracking during training.

    Args:
        example: Processed LiveCodeBench example
        idx: Index of the example
        data_source: Source dataset name
        ability: Ability category
        split: Dataset split (train/test)

    Returns:
        dict: Processed example with UID
    """
    question, test_cases = process_livecodebench_example(example)

    # Generate unique identifier for tracking
    unique_id = str(uuid.uuid4())

    data = {
        "uid": unique_id,
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": test_cases},
        "extra_info": {
            "split": split,
            "index": idx,
            "question_id": example.get("question_id", "unknown"),
            "contest_date": example.get("contest_date", "unknown"),
            "difficulty": example.get("difficulty", "unknown"),
            "platform": example.get("platform", "unknown"),
        },
    }
    return data


def process_livecodebench_example(example):
    """
    Process LiveCodeBench example into prompt and test cases format.

    Args:
        example: Raw LiveCodeBench example from jsonl

    Returns:
        tuple: (formatted_prompt, compressed_test_cases)
    """
    # Construct Query Prompt
    # From https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/prompts/code_generation.py#L140
    query_prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\nQuestion: {example['question_content']}\n\n"

    if example.get("starter_code", "").strip():
        query_prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n```python\n{example['starter_code']}\n```"
    else:
        query_prompt += (
            "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```"
        )

    # Construct test cases
    public_test_cases = json.loads(example["public_test_cases"])
    try:
        private_test_cases = json.loads(example["private_test_cases"])
    except Exception as e:
        print(f"Error loading private test cases: {e}")
        private_test_cases = json.loads(pickle.loads(zlib.decompress(base64.b64decode(example["private_test_cases"].encode("utf-8")))))

    full_test_cases = public_test_cases + private_test_cases

    metadata = json.loads(example["metadata"]) if example["metadata"] else {}
    test_cases = {
        "inputs": [t["input"] for t in full_test_cases],
        "outputs": [t["output"] for t in full_test_cases],
        "fn_name": metadata.get("func_name", None),
    }
    test_cases_compressed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")

    return query_prompt, test_cases_compressed


def process_local_livecodebench(jsonl_path, output_dir, train_ratio=0.8, date_filter=True):
    """
    Process local LiveCodeBench jsonl file with UID tracking.

    Args:
        jsonl_path: Path to local jsonl file
        output_dir: Directory to save processed dataset
        train_ratio: Ratio for train split (rest goes to test)
        date_filter: Whether to filter by date range
    """
    print(f"Loading LiveCodeBench from {jsonl_path}...")

    # Load jsonl file
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples from local file")

    # Filter by date range for R1 evaluation (2024-08 to 2025-01) if requested
    if date_filter:
        print("Filtering dataset by date range (2024-08 to 2025-01)...")
        filtered_examples = []
        for example in examples:
            contest_date = example.get("contest_date", "")
            if "2024-08-00T00:00:00" <= contest_date < "2025-01-00T00:00:00":
                filtered_examples.append(example)
        examples = filtered_examples
        print(f"After date filtering: {len(examples)} examples")

    if len(examples) == 0:
        print("Warning: No examples found after filtering. Processing all examples...")
        # Reload without filtering
        examples = []
        with open(jsonl_path) as f:
            for line in f:
                examples.append(json.loads(line))

    # Set random seed for reproducible splits
    random.seed(42)

    # Shuffle dataset indices
    indices = list(range(len(examples)))
    random.shuffle(indices)

    # Split indices
    train_size = int(len(indices) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    print(f"Split: {len(train_indices)} train, {len(test_indices)} test examples")

    # Use standard data_source for reward function recognition
    data_source = "livecodebench/code_generation_lite"

    # Process train split
    train_data = []
    for i, idx in enumerate(train_indices):
        example = examples[idx]
        processed = add_uid_to_example(example, i, data_source, "Code", "train")
        train_data.append(processed)

    # Process test split
    test_data = []
    for i, idx in enumerate(test_indices):
        example = examples[idx]
        processed = add_uid_to_example(example, i, data_source, "Code", "test")
        test_data.append(processed)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets
    train_path = os.path.join(output_dir, "train_with_uid.parquet")
    test_path = os.path.join(output_dir, "test_with_uid.parquet")

    # Convert to DataFrames and save
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Saved train dataset to: {train_path}")
    print(f"Saved test dataset to: {test_path}")

    # Print sample for verification
    print("\nSample train example:")
    sample = train_data[0]
    print(f"UID: {sample['uid']}")
    print(f"Data source: {sample['data_source']}")
    print(f"Question ID: {sample['extra_info']['question_id']}")
    print(f"Contest date: {sample['extra_info']['contest_date']}")
    print(f"Difficulty: {sample['extra_info']['difficulty']}")
    print(f"Platform: {sample['extra_info']['platform']}")
    print(f"Prompt preview: {sample['prompt'][0]['content'][:200]}...")

    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Process local LiveCodeBench jsonl file with UID tracking for verl training")
    parser.add_argument("--jsonl_path", default="/root/verl/liveCodebench_dataset/test5.jsonl", help="Path to local jsonl file")
    parser.add_argument("--output_dir", default="/root/verl/dataset/livecodebench", help="Directory to save processed dataset files")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data to use for training (default: 0.8)")
    parser.add_argument("--no_date_filter", action="store_true", help="Skip date filtering (use all examples)")

    args = parser.parse_args()

    print("Starting local LiveCodeBench dataset processing...")
    try:
        train_path, test_path = process_local_livecodebench(args.jsonl_path, args.output_dir, args.train_ratio, date_filter=not args.no_date_filter)
        print("Dataset processing completed successfully!")
        print("\nDataset ready for training:")
        print(f"  Train: {train_path}")
        print(f"  Test: {test_path}")
        print("\nTo run training, use:")
        print("  bash examples/split_placement/run_livecodebench_split.sh")
        return 0
    except Exception as e:
        print(f"Error during dataset processing: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
