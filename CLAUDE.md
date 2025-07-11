# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Installation and Setup
```bash
# Install main package
pip install -e .

# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks for linting
pre-commit install

# Install optional dependencies for specific engines
pip install -r requirements_sglang.txt  # For SGLang support
pip install -r requirements-npu.txt    # For NPU support
```

### Development Commands
```bash
# Run linting and formatting
pre-commit run --all-files

# Run E2E tests (requires GPU for full test suite)
cd tests/e2e && bash run_test.sh ppo_trainer vllm  # PPO with vLLM backend
cd tests/e2e && bash run_test.sh ppo_trainer sglang  # PPO with SGLang backend
cd tests/e2e && bash run_test.sh ppo_megatron_trainer  # PPO with Megatron backend

# Run specific test categories - CPU tests that don't require GPU
cd tests && python -m pytest ray_cpu/
cd tests && python -m pytest utils/cpu_tests/

# Run GPU tests (requires CUDA environment)
cd tests && python -m pytest utils/gpu_tests/
cd tests && python -m pytest ray_gpu/

# Run single test file
cd tests && python -m pytest utils/cpu_tests/test_import_utils.py

# Build documentation
cd docs && make html
```

### Running Training Examples
```bash
# PPO training example
python -m verl.trainer.main_ppo --config-name ppo_trainer

# GRPO training example  
bash examples/grpo_trainer/run_qwen2-7b.sh

# SFT training example
bash examples/sft/gsm8k/run_qwen_05_sp2.sh

# Evaluation
python -m verl.trainer.main_eval --config-name evaluation
```

### Working with UID Datasets
```bash
# Add UIDs to existing dataset
python /root/add_uid.py  # Processes GSM8K train/test parquet files

# Export UID dataset to CSV
python scripts/export_dataset_with_uid.py

# Process new dataset with UIDs (example workflow)
python examples/data_preprocess/gsm8k.py  # Creates GSM8K dataset
python /root/add_uid.py  # Adds UIDs for tracking
```

## Architecture Overview

### Core Components

**verl** is a reinforcement learning framework for large language models with a hybrid controller programming model:

- **Trainer Module** (`verl/trainer/`): Main entry points for training (PPO, GRPO, SFT)
- **Workers** (`verl/workers/`): Distributed training workers (actor, critic, rollout, reward)
- **Models** (`verl/models/`): Model integrations (Transformers, Megatron-LM, mcore)
- **Utils** (`verl/utils/`): Core utilities (datasets, checkpointing, distributed)

### Key Architectural Patterns

1. **Hybrid Controller Design**: Separates computation and data dependencies, enabling flexible device placement and seamless integration with existing LLM frameworks

2. **Multi-Backend Support**:
   - Training: FSDP, FSDP2, Megatron-LM
   - Inference: vLLM, SGLang, HF Transformers
   - Hardware: NVIDIA GPUs, AMD ROCm, Ascend NPUs

3. **Modular Workers**:
   - `actor/`: Policy model training workers
   - `critic/`: Value model workers  
   - `rollout/`: Generation/sampling workers
   - `reward_manager/`: Reward computation workers

4. **Config-Driven Training**: Hydra-based configuration system with YAML configs in `verl/trainer/config/`

### Model Integration Points

- **Transformers Models**: `verl/models/transformers/` - HuggingFace model wrappers
- **Megatron Integration**: `verl/models/*/megatron/` - Megatron-LM model implementations
- **Weight Loading**: `verl/models/weight_loader_registry.py` - Centralized weight loading

### Key Data Flow

1. **Data Loading**: `verl/utils/dataset/` handles RL datasets with prompt/response pairs
2. **Rollout Generation**: `verl/workers/rollout/` generates responses using inference engines
3. **Reward Computation**: `verl/workers/reward_manager/` computes rewards (model-based or function-based)
4. **Training**: `verl/trainer/ppo/` orchestrates PPO training loops

### Dataset Format

**GSM8K with UID**: The project uses GSM8K math dataset with unique identifiers for tracking:
- **Source**: Located in `/root/data/gsm8k/` with files like `train_with_uid.csv`, `test_with_uid.csv`
- **UID Column**: Each sample has a `uid` field with UUID for tracking individual samples during training
- **Structure**: Contains `data_source`, `prompt`, `ability`, `reward_model`, `extra_info`, and `uid` columns
- **Generation Script**: Use `/root/add_uid.py` to add UIDs to standard datasets
- **Export Script**: Use `scripts/export_dataset_with_uid.py` to convert datasets with UIDs to CSV format

### Core Configuration System

- **Main Config Files**: `verl/trainer/config/*.yaml` - Primary trainer configurations
- **Algorithm Examples**: `examples/{algorithm}_trainer/run_*.sh` - Executable examples with specific configs
- **Hydra Overrides**: Use `key=value` syntax to override any config parameter at runtime
- **Multi-Backend Support**: Same config format works across FSDP, FSDP2, and Megatron backends

## Development Guidelines

### Adding New Models
- Extend `verl/models/transformers/` for HF models
- Add Megatron variants in `verl/models/{model_name}/megatron/`
- Register in `verl/models/registry.py`

### Adding New RL Algorithms
- Follow `verl/trainer/ppo/` structure
- Implement core algorithms in `core_algos.py`
- Add ray trainer in `ray_trainer.py`

### Configuration Management
- Training configs in `verl/trainer/config/`
- Algorithm-specific configs in `examples/{algorithm}_trainer/`
- Use Hydra overrides for customization

### Testing Strategy
- **CPU Unit Tests**: `tests/ray_cpu/`, `tests/utils/cpu_tests/` - No GPU required
- **GPU Unit Tests**: `tests/ray_gpu/`, `tests/utils/gpu_tests/` - Requires CUDA
- **E2E Tests**: `tests/e2e/` - Full integration tests with minimal models (Qwen2.5-0.5B)
- **Component Tests**: Organized by worker type (`workers/`, `models/`, etc.)
- **Test Framework**: Uses pytest with Ray for distributed testing
- **Standardized E2E**: `tests/e2e/run_test.sh CONFIG_NAME [ENGINE]` pattern

### Performance Considerations
- Enable FSDP2 for better memory efficiency: `strategy=fsdp2`
- Use sequence packing for throughput: `balance_algo=reduce_latency`
- Leverage vLLM/SGLang for inference optimization
- See performance tuning guide in docs for detailed optimization

## Common Workflows

### Multi-turn Training with Tools
- Use SGLang backend: `examples/sglang_multiturn/`
- Tool calling support in `verl/tools/`
- See sandbox fusion examples for complex tool interactions

### Large Model Training (70B+)
- Use examples in `examples/tuning/{model_size}/`
- Leverage FSDP2 with CPU offloading
- Consider Megatron backend for very large models

### Custom Reward Functions
- Implement in `verl/utils/reward_score/`
- Support both model-based and verifiable rewards
- Math/coding verification in `prime_math/` and `prime_code/`

## Debugging and Troubleshooting

### Debug Commands
```bash
# Check Ray cluster status
ray status

# Monitor GPU usage during training
nvidia-smi -l 1

# Debug Ray workers
RAY_DEDUP_LOGS=0 python -m verl.trainer.main_ppo --config-name ppo_trainer

# Profile memory usage
python scripts/diagnose.py

# Check import issues
python -c "import verl; print('Import successful')"
```

### Common Issues
- **OOM Errors**: Try FSDP2 with CPU offloading: `actor_rollout_ref.actor.offload_policy=True`
- **Ray Connection Issues**: Check `ray status` and ensure cluster is properly initialized
- **vLLM Version Conflicts**: Use vLLM >= 0.8.2, avoid 0.7.x versions
- **Model Loading**: Use `weight_loader_registry.py` for custom weight loading patterns