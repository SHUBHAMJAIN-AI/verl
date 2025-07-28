#!/usr/bin/env python3
"""
Checkpoint Cleaner Script for VERL Training

This script automatically deletes old checkpoints while keeping the last N checkpoints
for fault tolerance. Runs continuously with configurable intervals.

Usage:
    python checkpoint_cleaner.py [options]

Example:
    python checkpoint_cleaner.py --base-dir /root/code/verl/checkpoints --keep-last 2 --interval 3600
"""

import os
import re
import time
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('checkpoint_cleaner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_checkpoint_dirs(base_dir: str) -> List[Tuple[str, List[Tuple[int, str]]]]:
    """
    Find all checkpoint directories and extract step numbers.
    
    Args:
        base_dir: Base directory to search for checkpoints
        
    Returns:
        List of tuples (experiment_path, [(step_number, checkpoint_path), ...])
    """
    base_path = Path(base_dir)
    experiments = []
    
    if not base_path.exists():
        logger.warning(f"Base directory {base_dir} does not exist")
        return experiments
    
    # Look for experiment directories (e.g., verl_gsm8k/ppo_run_27)
    for project_dir in base_path.iterdir():
        if not project_dir.is_dir():
            continue
            
        for exp_dir in project_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            # Find checkpoint directories with pattern global_step_N
            checkpoints = []
            checkpoint_pattern = re.compile(r'global_step_(\d+)$')
            
            for checkpoint_dir in exp_dir.iterdir():
                if not checkpoint_dir.is_dir():
                    continue
                    
                match = checkpoint_pattern.match(checkpoint_dir.name)
                if match:
                    step_number = int(match.group(1))
                    checkpoints.append((step_number, str(checkpoint_dir)))
            
            if checkpoints:
                # Sort by step number
                checkpoints.sort(key=lambda x: x[0])
                experiments.append((str(exp_dir), checkpoints))
                
    return experiments


def get_checkpoint_size(checkpoint_path: str) -> float:
    """Get total size of checkpoint directory in GB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(checkpoint_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except (OSError, FileNotFoundError):
                pass
    return total_size / (1024**3)  # Convert to GB


def cleanup_old_checkpoints(experiment_path: str, checkpoints: List[Tuple[int, str]], keep_last: int) -> None:
    """
    Clean up old checkpoints, keeping only the last N checkpoints.
    
    Args:
        experiment_path: Path to the experiment directory
        checkpoints: List of (step_number, checkpoint_path) tuples, sorted by step
        keep_last: Number of latest checkpoints to keep
    """
    if len(checkpoints) <= keep_last:
        logger.info(f"Experiment {experiment_path}: Only {len(checkpoints)} checkpoints found, keeping all")
        return
    
    # Calculate which checkpoints to delete
    to_delete = checkpoints[:-keep_last]
    to_keep = checkpoints[-keep_last:]
    
    logger.info(f"Experiment {experiment_path}:")
    logger.info(f"  Total checkpoints: {len(checkpoints)}")
    logger.info(f"  Keeping last {keep_last}: steps {[step for step, _ in to_keep]}")
    logger.info(f"  Deleting {len(to_delete)} old checkpoints: steps {[step for step, _ in to_delete]}")
    
    total_freed_space = 0.0
    
    for step_number, checkpoint_path in to_delete:
        try:
            # Calculate size before deletion
            size_gb = get_checkpoint_size(checkpoint_path)
            
            # Delete the checkpoint directory
            shutil.rmtree(checkpoint_path)
            total_freed_space += size_gb
            
            logger.info(f"  Deleted checkpoint step_{step_number} (freed {size_gb:.2f} GB)")
            
        except Exception as e:
            logger.error(f"  Failed to delete checkpoint {checkpoint_path}: {e}")
    
    logger.info(f"  Total space freed: {total_freed_space:.2f} GB")


def check_if_training_active(experiment_path: str, checkpoints: List[Tuple[int, str]]) -> bool:
    """
    Check if training is currently active by looking at recent checkpoint modification times.
    
    Args:
        experiment_path: Path to the experiment directory
        checkpoints: List of checkpoints
        
    Returns:
        True if training appears to be active (recent checkpoint modifications)
    """
    if not checkpoints:
        return False
    
    # Check the most recent checkpoint's modification time
    latest_step, latest_checkpoint = checkpoints[-1]
    
    try:
        # Check modification time of the most recent checkpoint directory
        mtime = os.path.getmtime(latest_checkpoint)
        time_since_modified = time.time() - mtime
        
        # Consider training active if last checkpoint was modified within 4 hours
        is_active = time_since_modified < 14400  # 4 hours in seconds
        
        if is_active:
            logger.info(f"Training appears active in {experiment_path} (last checkpoint modified {time_since_modified/60:.1f} minutes ago)")
        else:
            logger.info(f"Training appears inactive in {experiment_path} (last checkpoint modified {time_since_modified/3600:.1f} hours ago)")
            
        return is_active
        
    except OSError:
        logger.warning(f"Could not check modification time for {latest_checkpoint}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Clean up old VERL training checkpoints')
    parser.add_argument('--base-dir', 
                        default='/root/code/verl/checkpoints',
                        help='Base directory containing checkpoints (default: /root/code/verl/checkpoints)')
    parser.add_argument('--keep-last', 
                        type=int, 
                        default=3,
                        help='Number of latest checkpoints to keep (default: 3)')
    parser.add_argument('--interval', 
                        type=int, 
                        default=10800,
                        help='Cleanup interval in seconds (default: 10800 = 3 hours)')
    parser.add_argument('--once', 
                        action='store_true',
                        help='Run cleanup once and exit (don\'t run continuously)')
    parser.add_argument('--only-active', 
                        action='store_true',
                        help='Only clean checkpoints from currently active training runs')
    parser.add_argument('--dry-run', 
                        action='store_true',
                        help='Show what would be deleted without actually deleting')
    
    args = parser.parse_args()
    
    logger.info("=== VERL Checkpoint Cleaner Started ===")
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Keep last: {args.keep_last} checkpoints")
    logger.info(f"Interval: {args.interval} seconds ({args.interval/3600:.1f} hours)")
    logger.info(f"Run once: {args.once}")
    logger.info(f"Only active training: {args.only_active}")
    logger.info(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        logger.warning("DRY RUN MODE - No files will actually be deleted")
    
    while True:
        try:
            logger.info(f"\n--- Cleanup run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            # Find all experiments with checkpoints
            experiments = find_checkpoint_dirs(args.base_dir)
            
            if not experiments:
                logger.info("No checkpoint directories found")
            else:
                logger.info(f"Found {len(experiments)} experiments with checkpoints")
                
                for experiment_path, checkpoints in experiments:
                    # Check if we should only process active training runs
                    if args.only_active:
                        if not check_if_training_active(experiment_path, checkpoints):
                            logger.info(f"Skipping inactive experiment: {experiment_path}")
                            continue
                    
                    if not args.dry_run:
                        cleanup_old_checkpoints(experiment_path, checkpoints, args.keep_last)
                    else:
                        # Dry run - just show what would be deleted
                        if len(checkpoints) > args.keep_last:
                            to_delete = checkpoints[:-args.keep_last]
                            logger.info(f"DRY RUN - Would delete {len(to_delete)} checkpoints from {experiment_path}:")
                            for step, path in to_delete:
                                size_gb = get_checkpoint_size(path)
                                logger.info(f"  Would delete step_{step} ({size_gb:.2f} GB)")
            
            if args.once:
                logger.info("Single run completed, exiting")
                break
                
            logger.info(f"Cleanup completed, sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"Unexpected error during cleanup: {e}")
            if args.once:
                break
            else:
                logger.info(f"Continuing after error, sleeping for {args.interval} seconds...")
                time.sleep(args.interval)
    
    logger.info("=== VERL Checkpoint Cleaner Stopped ===")


if __name__ == "__main__":
    main()