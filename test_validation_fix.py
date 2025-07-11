#!/usr/bin/env python3
"""
Test script to verify that test validation files are created separately
"""
import os
import tempfile
from unittest.mock import Mock, MagicMock
import sys

# Add the verl path
sys.path.insert(0, '/root/code/verl')

def test_separate_validation_files():
    # Mock the required modules
    mock_wandb = MagicMock()
    mock_wandb.run = True
    mock_wandb.Table = MagicMock()
    mock_wandb.Artifact = MagicMock()
    
    # Patch wandb in metric_utils
    import verl.trainer.ppo.metric_utils as metric_utils
    metric_utils.wandb = mock_wandb
    metric_utils.WANDB_AVAILABLE = True
    
    # Create a mock batch for testing
    mock_batch = Mock()
    mock_batch.batch = {
        'prompts': ['test prompt'],
        'responses_text': ['test response'],
        'uid': ['test-uid-123']
    }
    mock_batch.non_tensor_batch = {}
    
    # Mock torch tensors for lengths
    import torch
    def mock_compute_response_info(batch):
        return {
            'prompt_length': torch.tensor([10]),
            'response_length': torch.tensor([20])
        }
    
    # Patch the internal function
    original_compute = metric_utils._compute_response_info
    metric_utils._compute_response_info = mock_compute_response_info
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Testing in temporary directory: {temp_dir}")
            
            # Test training step (should create individual_lengths_step_X.csv)
            metric_utils.log_individual_lengths_to_wandb_and_csv(
                mock_batch,
                step=5,
                output_dir=temp_dir,
                universal_id_key="uid"
            )
            
            # Test validation step (should create test_validation_step_X.csv)
            metric_utils.log_individual_lengths_to_wandb_and_csv(
                mock_batch,
                step="test_step_10",
                output_dir=temp_dir,
                universal_id_key="uid"
            )
            
            # Check files were created
            files = os.listdir(temp_dir)
            print(f"Created files: {files}")
            
            # Verify correct file names
            training_file = f"individual_lengths_step_5.csv"
            validation_file = f"test_validation_step_10.csv"
            
            assert training_file in files, f"Training file {training_file} not found"
            assert validation_file in files, f"Validation file {validation_file} not found"
            
            print("✅ SUCCESS: Separate validation files are created correctly!")
            print(f"  - Training: {training_file}")
            print(f"  - Validation: {validation_file}")
            
            # Check file contents
            import pandas as pd
            
            validation_df = pd.read_csv(os.path.join(temp_dir, validation_file))
            print(f"\nValidation file columns: {list(validation_df.columns)}")
            print(f"Validation file shape: {validation_df.shape}")
            
    finally:
        # Restore original function
        metric_utils._compute_response_info = original_compute

if __name__ == "__main__":
    test_separate_validation_files()