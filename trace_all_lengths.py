#!/usr/bin/env python3
"""
Script to trace prompt and response lengths for all samples including:
1. Missing 98 training records from RL training (2 epochs)
2. Complete test set analysis (currently missing)
"""

import pandas as pd
import json
import ast
from pathlib import Path

def count_tokens_in_sequence(token_sequence):
    """Count tokens in a token sequence"""
    if isinstance(token_sequence, str):
        try:
            if token_sequence.startswith('[') and token_sequence.endswith(']'):
                token_list = ast.literal_eval(token_sequence)
                return len(token_list)
        except (ValueError, SyntaxError):
            return len(token_sequence.split(','))
    elif isinstance(token_sequence, list):
        return len(token_sequence)
    return 0

def analyze_gsm8k_dataset(file_path, dataset_name):
    """Analyze GSM8K dataset and extract prompt/response lengths"""
    print(f"Analyzing {dataset_name}: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")
    
    results = []
    
    for idx, row in df.iterrows():
        uid = row['uid']
        
        # Parse prompt (JSON format)
        prompt_text = ""
        prompt_tokens = 0
        if pd.notna(row['prompt']):
            try:
                # Handle string representation of list with dict
                prompt_str = str(row['prompt'])
                if prompt_str.startswith("[{") and prompt_str.endswith("}]"):
                    # Replace single quotes with double quotes for proper JSON
                    prompt_str = prompt_str.replace("'", '"')
                    prompt_data = json.loads(prompt_str)
                    if isinstance(prompt_data, list) and len(prompt_data) > 0:
                        prompt_text = prompt_data[0].get('content', '')
                        prompt_tokens = len(prompt_text.split())
            except json.JSONDecodeError:
                # Try using ast.literal_eval for Python list strings
                try:
                    prompt_data = ast.literal_eval(str(row['prompt']))
                    if isinstance(prompt_data, list) and len(prompt_data) > 0:
                        prompt_text = prompt_data[0].get('content', '')
                        prompt_tokens = len(prompt_text.split())
                except (ValueError, SyntaxError):
                    prompt_text = str(row['prompt'])
                    prompt_tokens = len(prompt_text.split())
        
        # Parse response from extra_info
        response_text = ""
        response_tokens = 0
        if pd.notna(row['extra_info']):
            try:
                # Handle string representation of dict
                extra_info_str = str(row['extra_info'])
                if extra_info_str.startswith("{") and extra_info_str.endswith("}"):
                    # Replace single quotes with double quotes for proper JSON
                    extra_info_str = extra_info_str.replace("'", '"')
                    extra_info = json.loads(extra_info_str)
                    response_text = extra_info.get('answer', '')
                    response_tokens = len(response_text.split())
            except json.JSONDecodeError:
                # Try using ast.literal_eval for Python dict strings
                try:
                    extra_info = ast.literal_eval(str(row['extra_info']))
                    response_text = extra_info.get('answer', '')
                    response_tokens = len(response_text.split())
                except (ValueError, SyntaxError):
                    pass
        
        results.append({
            'uid': uid,
            'dataset': dataset_name,
            'index': idx,
            'data_source': row.get('data_source', ''),
            'ability': row.get('ability', ''),
            'prompt_text': prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text,
            'response_text': response_text[:200] + "..." if len(response_text) > 200 else response_text,
            'prompt_length': prompt_tokens,
            'response_length': response_tokens,
            'total_length': prompt_tokens + response_tokens
        })
    
    return pd.DataFrame(results)

def analyze_missing_training_records():
    """Find which training records are missing from RL outputs"""
    print("Analyzing missing training records...")
    
    # Load original training data
    train_df = pd.read_csv('/root/data/gsm8k/train_with_uid.csv')
    train_uids = set(train_df['uid'])
    print(f"Original training UIDs: {len(train_uids)}")
    
    # Load RL training outputs
    outputs_df = pd.read_csv('/root/code/verl/outputs/combined.csv')
    tracked_uids = set(outputs_df['universal_id'])
    print(f"Tracked UIDs in RL outputs: {len(tracked_uids)}")
    
    # Find missing UIDs
    missing_uids = train_uids - tracked_uids
    print(f"Missing UIDs: {len(missing_uids)}")
    
    if missing_uids:
        # Get details of missing records
        missing_records = train_df[train_df['uid'].isin(missing_uids)]
        print(f"Missing records details:")
        for idx, row in missing_records.iterrows():
            print(f"  UID: {row['uid']}, Index: {idx}")
        
        # Save missing records analysis
        missing_analysis = analyze_gsm8k_dataset('/root/data/gsm8k/train_with_uid.csv', 'train')
        missing_subset = missing_analysis[missing_analysis['uid'].isin(missing_uids)]
        missing_subset.to_csv('/root/code/verl/outputs/missing_training_records.csv', index=False)
        print(f"Missing training records saved to: /root/code/verl/outputs/missing_training_records.csv")
        
        return missing_subset
    
    return pd.DataFrame()

def create_test_set_analysis():
    """Create complete test set length analysis"""
    print("Creating test set analysis...")
    
    test_analysis = analyze_gsm8k_dataset('/root/data/gsm8k/test_with_uid.csv', 'test')
    test_analysis.to_csv('/root/code/verl/outputs/test_set_lengths.csv', index=False)
    print(f"Test set analysis saved to: /root/code/verl/outputs/test_set_lengths.csv")
    
    return test_analysis

def create_complete_training_analysis():
    """Create complete training set analysis"""
    print("Creating complete training set analysis...")
    
    train_analysis = analyze_gsm8k_dataset('/root/data/gsm8k/train_with_uid.csv', 'train')
    train_analysis.to_csv('/root/code/verl/outputs/complete_training_lengths.csv', index=False)
    print(f"Complete training analysis saved to: /root/code/verl/outputs/complete_training_lengths.csv")
    
    return train_analysis

def generate_summary_report(train_df, test_df, missing_df):
    """Generate summary report"""
    
    report = []
    report.append("DATASET LENGTH TRACKING SUMMARY")
    report.append("=" * 40)
    report.append("")
    
    # Training dataset
    if not train_df.empty:
        report.append("TRAINING DATASET:")
        report.append(f"  Total samples: {len(train_df)}")
        report.append(f"  Avg prompt length: {train_df['prompt_length'].mean():.1f} words")
        report.append(f"  Avg response length: {train_df['response_length'].mean():.1f} words")
        report.append(f"  Avg total length: {train_df['total_length'].mean():.1f} words")
        report.append("")
    
    # Test dataset
    if not test_df.empty:
        report.append("TEST DATASET:")
        report.append(f"  Total samples: {len(test_df)}")
        report.append(f"  Avg prompt length: {test_df['prompt_length'].mean():.1f} words")
        report.append(f"  Avg response length: {test_df['response_length'].mean():.1f} words")
        report.append(f"  Avg total length: {test_df['total_length'].mean():.1f} words")
        report.append("")
    
    # RL Training Coverage
    outputs_df = pd.read_csv('/root/code/verl/outputs/combined.csv')
    report.append("RL TRAINING COVERAGE:")
    report.append(f"  Expected records (2 epochs): {len(train_df) * 2}")
    report.append(f"  Actual records tracked: {len(outputs_df)}")
    report.append(f"  Missing records: {(len(train_df) * 2) - len(outputs_df)}")
    report.append(f"  Coverage: {(len(outputs_df) / (len(train_df) * 2)) * 100:.1f}%")
    report.append("")
    
    # Missing records
    if not missing_df.empty:
        report.append("MISSING TRAINING RECORDS:")
        report.append(f"  Count: {len(missing_df)}")
        report.append("  UIDs:")
        for uid in missing_df['uid']:
            report.append(f"    {uid}")
        report.append("")
    
    report.append("FILES GENERATED:")
    report.append("  - complete_training_lengths.csv: All training samples with lengths")
    report.append("  - test_set_lengths.csv: All test samples with lengths")
    report.append("  - missing_training_records.csv: Training records not in RL outputs")
    report.append("  - summary_report.txt: This report")
    
    # Save report
    with open('/root/code/verl/outputs/summary_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))

def main():
    print("Starting comprehensive dataset length analysis...")
    
    # Create complete training analysis
    train_df = create_complete_training_analysis()
    
    # Create test set analysis (currently missing)
    test_df = create_test_set_analysis()
    
    # Find missing training records
    missing_df = analyze_missing_training_records()
    
    # Generate summary report
    generate_summary_report(train_df, test_df, missing_df)
    
    print("\nAnalysis complete! Check /root/code/verl/outputs/ for results.")

if __name__ == "__main__":
    main()