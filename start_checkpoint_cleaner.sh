#!/bin/bash
# Convenience script to start the checkpoint cleaner with common configurations

# Default configuration - modify these as needed
BASE_DIR="/root/code/verl/checkpoints"
KEEP_LAST=3
INTERVAL=10800  # 3 hours in seconds
ONLY_ACTIVE=true

echo "Starting VERL Checkpoint Cleaner..."
echo "Base directory: $BASE_DIR"
echo "Keep last: $KEEP_LAST checkpoints"
echo "Cleanup interval: $INTERVAL seconds ($(($INTERVAL / 3600)) hours)"
echo "Only active training: $ONLY_ACTIVE"
echo ""

# Activate conda environment if needed
if [[ "$CONDA_DEFAULT_ENV" != "verl_new" ]]; then
    echo "Activating verl_new conda environment..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate verl_new
fi

# Run the checkpoint cleaner
if [[ "$ONLY_ACTIVE" == "true" ]]; then
    echo "Running with --only-active flag..."
    python /root/code/verl/checkpoint_cleaner.py \
        --base-dir "$BASE_DIR" \
        --keep-last "$KEEP_LAST" \
        --interval "$INTERVAL" \
        --only-active
else
    echo "Running without activity detection..."
    python /root/code/verl/checkpoint_cleaner.py \
        --base-dir "$BASE_DIR" \
        --keep-last "$KEEP_LAST" \
        --interval "$INTERVAL"
fi