import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

def create_response_length_curves():
    print("Loading combined individual lengths data...")
    df = pd.read_csv('/root/code/verl/outputs/combined_individual_lengths.csv')
    
    print(f"Data shape: {df.shape}")
    print(f"Steps range: {df['step'].min()} to {df['step'].max()}")
    print(f"Response length range: {df['response_length'].min()} to {df['response_length'].max()}")
    
    # Convert steps to epochs (59 steps = 1 epoch)
    df['epoch'] = df['step'] / 59.0
    
    # Group by universal_id to get individual prompt trajectories
    prompt_trajectories = defaultdict(list)
    
    for _, row in df.iterrows():
        prompt_id = row['universal_id']
        epoch = row['epoch']
        response_length = row['response_length']
        prompt_trajectories[prompt_id].append((epoch, response_length))
    
    # Sort trajectories by epoch for each prompt
    for prompt_id in prompt_trajectories:
        prompt_trajectories[prompt_id].sort(key=lambda x: x[0])
    
    # Filter trajectories that have sufficient data points (at least 10 epochs)
    filtered_trajectories = {pid: traj for pid, traj in prompt_trajectories.items() 
                           if len(traj) >= 10}
    
    print(f"Found {len(filtered_trajectories)} prompts with sufficient data points")
    
    # Select 20 diverse curves
    # Sort by various criteria to get diversity
    trajectory_stats = {}
    for pid, traj in filtered_trajectories.items():
        lengths = [point[1] for point in traj]
        trajectory_stats[pid] = {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'length_range': np.max(lengths) - np.min(lengths),
            'trajectory': traj
        }
    
    # Select 20 diverse trajectories
    selected_trajectories = []
    
    # Group by different characteristics to ensure diversity
    by_mean_length = sorted(trajectory_stats.items(), key=lambda x: x[1]['mean_length'])
    by_variability = sorted(trajectory_stats.items(), key=lambda x: x[1]['std_length'])
    by_range = sorted(trajectory_stats.items(), key=lambda x: x[1]['length_range'])
    
    # Take samples from different groups
    selected_pids = set()
    
    # Low, medium, high mean lengths
    for i in [0, len(by_mean_length)//4, len(by_mean_length)//2, 3*len(by_mean_length)//4, -1]:
        if i < len(by_mean_length):
            selected_pids.add(by_mean_length[i][0])
    
    # Low, medium, high variability
    for i in [0, len(by_variability)//4, len(by_variability)//2, 3*len(by_variability)//4, -1]:
        if i < len(by_variability):
            selected_pids.add(by_variability[i][0])
    
    # Low, medium, high range
    for i in [0, len(by_range)//4, len(by_range)//2, 3*len(by_range)//4, -1]:
        if i < len(by_range):
            selected_pids.add(by_range[i][0])
    
    # Fill remaining with random selection
    remaining_pids = list(set(trajectory_stats.keys()) - selected_pids)
    np.random.seed(42)
    additional_pids = np.random.choice(remaining_pids, 
                                     size=min(20 - len(selected_pids), len(remaining_pids)), 
                                     replace=False)
    selected_pids.update(additional_pids)
    
    selected_trajectories = [(pid, trajectory_stats[pid]['trajectory']) 
                           for pid in list(selected_pids)[:20]]
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Use a colormap for diverse colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(selected_trajectories)))
    
    for i, (pid, trajectory) in enumerate(selected_trajectories):
        epochs = [point[0] for point in trajectory]
        lengths = [point[1] for point in trajectory]
        
        plt.plot(epochs, lengths, 
                color=colors[i], 
                alpha=0.7, 
                linewidth=1.5,
                label=f'Prompt {i+1}')
    
    plt.xlabel('Training Epoch', fontsize=12)
    plt.ylabel('Response Length (tokens)', fontsize=12)
    plt.title('Response Length Evolution Across Training Epochs\n20 Diverse Prompt Trajectories', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend with smaller font
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('/root/code/verl/response_length_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics about selected trajectories
    print("\nSelected Trajectory Statistics:")
    for i, (pid, trajectory) in enumerate(selected_trajectories):
        lengths = [point[1] for point in trajectory]
        stats = trajectory_stats[pid]
        print(f"Prompt {i+1}: Mean={stats['mean_length']:.1f}, "
              f"Std={stats['std_length']:.1f}, "
              f"Range={stats['min_length']}-{stats['max_length']}")

if __name__ == "__main__":
    create_response_length_curves()