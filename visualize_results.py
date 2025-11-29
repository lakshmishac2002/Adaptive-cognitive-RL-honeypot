#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np

def visualize():
    print(" Creating visualizations...")
    
    # Load training history
    with open('results/training_history_enhanced.json', 'r') as f:
        training_data = json.load(f)
    
    # Load evaluation results
    with open('results/evaluation_results.json', 'r') as f:
        eval_data = json.load(f)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Training Rewards Over Time
    ax1 = plt.subplot(2, 3, 1)
    rewards = training_data['episode_rewards']
    episodes = range(1, len(rewards) + 1)
    
    # Plot with moving average
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(rewards) + 1), moving_avg, 
                color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward Distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(rewards, bins=50, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance by Attack Type
    ax3 = plt.subplot(2, 3, 3)
    attack_perf = eval_data['attack_performance']
    attacks = list(attack_perf.keys())
    scores = list(attack_perf.values())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(attacks)))
    bars = ax3.barh(attacks, scores, color=colors, edgecolor='black')
    ax3.set_xlabel('Average Reward')
    ax3.set_title('Performance by Attack Type')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Action Distribution
    ax4 = plt.subplot(2, 3, 4)
    actions = list(eval_data['action_distribution'].keys())
    counts = list(eval_data['action_distribution'].values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(actions)))
    wedges, texts, autotexts = ax4.pie(counts, labels=actions, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
    ax4.set_title('Action Distribution')
    
    # 5. Learning Curve (Improvement over time)
    ax5 = plt.subplot(2, 3, 5)
    chunk_size = 100
    chunks = [rewards[i:i+chunk_size] for i in range(0, len(rewards), chunk_size)]
    chunk_avgs = [np.mean(chunk) for chunk in chunks]
    chunk_episodes = range(chunk_size, len(rewards) + 1, chunk_size)
    
    ax5.plot(chunk_episodes, chunk_avgs, marker='o', linewidth=2, 
            markersize=8, color='purple')
    ax5.set_xlabel(f'Episode (per {chunk_size})')
    ax5.set_ylabel('Average Reward')
    ax5.set_title(f'Learning Curve ({chunk_size}-Episode Chunks)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Training vs Evaluation Comparison
    ax6 = plt.subplot(2, 3, 6)
    
    train_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    eval_avg = eval_data['avg_reward']
    
    comparison = ['Training\n(Last 100)', 'Evaluation\n(100 episodes)']
    values = [train_avg, eval_avg]
    colors_comp = ['skyblue', 'lightcoral']
    
    bars = ax6.bar(comparison, values, color=colors_comp, edgecolor='black', width=0.6)
    ax6.set_ylabel('Average Reward')
    ax6.set_title('Training vs Evaluation Performance')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/training_visualization.png', dpi=300, bbox_inches='tight')
    print(" Saved: results/training_visualization.png")
    
    plt.show()

if __name__ == '__main__':
    visualize()
