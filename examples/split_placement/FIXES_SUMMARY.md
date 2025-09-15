# Split PPO vs Regular PPO - Issues Fixed

## Problems Identified:

### 1. **Missing Validation Runs (Primary Issue)**
- **Problem**: Split PPO had `test_freq: -1` → no validation during training
- **Fix**: Changed to `test_freq: 10` to match regular PPO
- **Impact**: Without validation, training accuracy remained 0.000 because no actual accuracy measurement was occurring

### 2. **Learning Rate Scheduler Issues**
- **Problem**: Split PPO had `lr_warmup_steps: 0` instead of `-1`
- **Fix**: Changed to `lr_warmup_steps: -1` (delegates to lr_warmup_steps_ratio)
- **Added**: `weight_decay: 0.01` for both actor and critic
- **Impact**: Should fix the 0.000 learning rate issue seen in logs

### 3. **Reward System Configuration**
- **Problem**: Missing custom_reward_function section
- **Fix**: Added custom_reward_function configuration block
- **Note**: Both configs use function-based rewards with data_source='openai/gsm8k'

### 4. **Configuration Alignment**
- **Added**: WandB logging to split PPO config
- **Added**: `log_val_generations: 5` for better debugging
- **Fixed**: `total_training_steps: -1` (program override) instead of hardcoded 1000

## Files Modified:

1. `/root/verl/examples/split_placement/config/qwen_0.5b_2gpu_split.yaml`
   - test_freq: -1 → 10
   - lr_warmup_steps: 0 → -1
   - total_training_steps: 1000 → -1
   - Added custom_reward_function section
   - Added weight_decay settings
   - Enabled WandB logging
   - Added validation generation logging

2. Created new run script: `run_qwen_0.5b_2gpu_fixed_v2.sh`
   - Includes all fixes and proper experiment naming

## Expected Results After Fixes:

### Split PPO should now show:
- **Validation runs every 10 steps** with metrics like `val-core/openai/gsm8k/reward/mean@1`
- **Non-zero learning rates** during training (5e-6 for actor, 1e-5 for critic)  
- **Progressive accuracy improvement** (though potentially lower than 3B model due to smaller size)
- **Meaningful reward values** instead of all 0.000

### Comparison will be fairer:
- Same dataset (GSM8K with UIDs)
- Same validation frequency 
- Same learning rate scheduler approach
- Proper WandB logging for both runs

## Key Differences Remaining:

1. **Model Size**: Regular PPO uses 3.09B parameters, Split PPO uses 494.03M parameters
2. **Hardware Setup**: Split PPO uses 2 GPUs with split placement, Regular PPO uses distributed setup
3. **Batch Sizes**: Different due to memory constraints on 2-GPU setup

## Next Steps:

1. Run the fixed script: `./run_qwen_0.5b_2gpu_fixed_v2.sh`
2. Monitor that validation runs occur every 10 steps
3. Verify learning rates are non-zero
4. Compare training curves between regular and split PPO
5. For fairer comparison, consider running regular PPO with same 0.5B model size