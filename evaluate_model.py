#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for RL Honeypot Agent
Tests the trained agent on various metrics and scenarios
"""
import torch
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from honeypot_environment_enhanced import EnhancedHoneypotEnvironment
from custom_rl_agent import FixedHoneypotRLAgent


class ModelEvaluator:
    """Comprehensive evaluation for trained RL honeypot agent"""
    
    def __init__(self, model_path, num_eval_episodes=100):
        self.model_path = model_path
        self.num_eval_episodes = num_eval_episodes
        
        # Initialize environment and agent
        self.env = EnhancedHoneypotEnvironment()
        self.agent = FixedHoneypotRLAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )
        
        # Load model with PyTorch 2.6+ compatibility
        checkpoint = torch.load(model_path, map_location=self.agent.device, weights_only=False)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.agent.target_net.load_state_dict(checkpoint['target_net'])
        self.agent.training_step = checkpoint['training_step']
        self.agent.epsilon = 0.0  # Set to 0 for evaluation (greedy policy)
        
        print(f" Model loaded from {model_path}")
        print(f"   Training steps: {self.agent.training_step}")
        print(f"   Evaluation mode: epsilon = {self.agent.epsilon}")
        print()
        
        # Results storage
        self.results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'action_distribution': defaultdict(int),
            'attack_type_performance': defaultdict(list),
            'action_per_attack': defaultdict(lambda: defaultdict(int)),
            'q_value_stats': [],
            'engagement_times': []
        }
    
    def evaluate(self):
        """Run comprehensive evaluation"""
        print("=" * 70)
        print("STARTING COMPREHENSIVE EVALUATION")
        print("=" * 70)
        print(f"Number of episodes: {self.num_eval_episodes}")
        print(f"Model: {self.model_path}")
        print()
        
        for episode in range(self.num_eval_episodes):
            episode_reward, episode_length, actions_taken = self._evaluate_episode()
            
            self.results['episode_rewards'].append(episode_reward)
            self.results['episode_lengths'].append(episode_length)
            self.results['engagement_times'].append(episode_length)
            
            # Track actions
            for action in actions_taken:
                self.results['action_distribution'][action] += 1
            
            # Progress indicator
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.results['episode_rewards'][-10:])
                avg_length = np.mean(self.results['episode_lengths'][-10:])
                print(f"Episode {episode + 1:3d}/{self.num_eval_episodes} | "
                      f"Avg Reward (last 10): {avg_reward:7.2f} | "
                      f"Avg Length: {avg_length:.1f} steps")
        
        print()
        print(" Evaluation complete!")
        print()
        
        # Generate report
        self._generate_report()
        
        # Save results
        self._save_results()
        
        # Create visualizations
        self._create_visualizations()
        
        return self.results
    
    def _evaluate_episode(self):
        """Evaluate a single episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        actions_taken = []
        done = False
        
        attack_type = self.env.attack_types[self.env.current_attack_type]
        
        while not done:
            # Get action from policy (greedy)
            action = self.agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Record
            actions_taken.append(action)
            episode_reward += reward
            episode_length += 1
            
            # Track action per attack type
            self.results['action_per_attack'][attack_type][action] += 1
            
            state = next_state
        
        # Store attack performance
        self.results['attack_type_performance'][attack_type].append(episode_reward)
        
        return episode_reward, episode_length, actions_taken
    
    def _generate_report(self):
        """Generate comprehensive evaluation report"""
        print("=" * 70)
        print("EVALUATION REPORT")
        print("=" * 70)
        print()
        
        # 1. Overall Performance
        print(" OVERALL PERFORMANCE")
        print("-" * 70)
        rewards = self.results['episode_rewards']
        print(f"Average Reward:     {np.mean(rewards):8.2f} ± {np.std(rewards):.2f}")
        print(f"Median Reward:      {np.median(rewards):8.2f}")
        print(f"Min/Max Reward:     {np.min(rewards):8.2f} / {np.max(rewards):.2f}")
        print(f"Average Episode Length: {np.mean(self.results['episode_lengths']):.1f} steps")
        print()
        
        # 2. Action Diversity
        print(" ACTION DIVERSITY")
        print("-" * 70)
        total_actions = sum(self.results['action_distribution'].values())
        action_names = {
            0: 'respond_realistic',
            1: 'delay_response',
            2: 'fake_error',
            3: 'fake_data',
            4: 'redirect_deep',
            5: 'block'
        }
        
        action_percentages = []
        for action_id in range(self.env.action_dim):
            count = self.results['action_distribution'][action_id]
            pct = (count / total_actions) * 100 if total_actions > 0 else 0
            action_percentages.append(pct)
            action_name = action_names.get(action_id, f"Action {action_id}")
            print(f"{action_name:20s}: {count:5d} times ({pct:5.1f}%)")
        
        # Calculate entropy
        probs = np.array(action_percentages) / 100.0
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0
        max_entropy = np.log(self.env.action_dim)
        diversity_pct = (entropy / max_entropy) * 100
        
        print()
        print(f"Entropy:            {entropy:.3f} / {max_entropy:.3f}")
        print(f"Diversity Score:    {diversity_pct:.1f}%")
        
        if diversity_pct > 75:
            print(" EXCELLENT diversity!")
        elif diversity_pct > 60:
            print(" Good diversity")
        else:
            print("  Low diversity")
        
        print()
        
        # 3. Performance by Attack Type
        print(" PERFORMANCE BY ATTACK TYPE")
        print("-" * 70)
        attack_stats = []
        for attack_type, rewards in sorted(self.results['attack_type_performance'].items()):
            if rewards:
                avg = np.mean(rewards)
                std = np.std(rewards)
                count = len(rewards)
                attack_stats.append({
                    'attack': attack_type,
                    'avg': avg,
                    'std': std,
                    'count': count
                })
                print(f"{attack_type:20s}: {avg:8.2f} ± {std:6.2f} ({count:3d} episodes)")
        
        print()
        
        # 4. Strategy Analysis (Actions per Attack Type)
        print(" STRATEGY ANALYSIS - Actions Used Per Attack Type")
        print("-" * 70)
        for attack_type in sorted(self.results['action_per_attack'].keys()):
            action_counts = self.results['action_per_attack'][attack_type]
            total = sum(action_counts.values())
            
            print(f"\n{attack_type}:")
            for action_id, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total) * 100 if total > 0 else 0
                action_name = action_names.get(action_id, f"Action {action_id}")
                if pct > 5:  # Only show actions used >5%
                    print(f"  {action_name:20s}: {pct:5.1f}%")
        
        print()
        
        # 5. Engagement Metrics
        print("  ENGAGEMENT METRICS")
        print("-" * 70)
        engagement_times = self.results['engagement_times']
        print(f"Average Engagement:  {np.mean(engagement_times):.1f} steps")
        print(f"Median Engagement:   {np.median(engagement_times):.1f} steps")
        print(f"Max Engagement:      {np.max(engagement_times):.1f} steps")
        print()
        
        # 6. Consistency Score
        print("📈 CONSISTENCY METRICS")
        print("-" * 70)
        reward_cv = np.std(rewards) / np.mean(rewards) if np.mean(rewards) != 0 else 0
        print(f"Reward Coefficient of Variation: {reward_cv:.3f}")
        if reward_cv < 0.3:
            print(" Very consistent performance")
        elif reward_cv < 0.5:
            print("✓ Moderately consistent")
        else:
            print("  High variance in performance")
        
        print()
        print("=" * 70)
    
    def _save_results(self):
        """Save evaluation results to JSON"""
        save_data = {
            'model_path': self.model_path,
            'num_episodes': self.num_eval_episodes,
            'timestamp': datetime.now().isoformat(),
            'overall_performance': {
                'mean_reward': float(np.mean(self.results['episode_rewards'])),
                'std_reward': float(np.std(self.results['episode_rewards'])),
                'median_reward': float(np.median(self.results['episode_rewards'])),
                'min_reward': float(np.min(self.results['episode_rewards'])),
                'max_reward': float(np.max(self.results['episode_rewards'])),
                'mean_episode_length': float(np.mean(self.results['episode_lengths']))
            },
            'action_distribution': {
                k: int(v) for k, v in self.results['action_distribution'].items()
            },
            'attack_type_performance': {
                attack: {
                    'mean': float(np.mean(rewards)),
                    'std': float(np.std(rewards)),
                    'count': len(rewards)
                }
                for attack, rewards in self.results['attack_type_performance'].items()
            }
        }
        
        filename = f"results/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f" Results saved to: {filename}")
    
    def _create_visualizations(self):
        """Create evaluation visualizations"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('RL Honeypot Agent - Evaluation Results', fontsize=16, fontweight='bold')
            
            # 1. Reward Distribution
            axes[0, 0].hist(self.results['episode_rewards'], bins=30, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(np.mean(self.results['episode_rewards']), color='red', 
                              linestyle='--', label=f"Mean: {np.mean(self.results['episode_rewards']):.1f}")
            axes[0, 0].set_xlabel('Episode Reward')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Reward Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Action Distribution
            action_names = ['respond_realistic', 'delay_response', 'fake_error', 
                           'fake_data', 'redirect_deep', 'block']
            action_counts = [self.results['action_distribution'][i] for i in range(6)]
            colors = plt.cm.Set3(range(6))
            axes[0, 1].bar(action_names, action_counts, color=colors, edgecolor='black')
            axes[0, 1].set_xlabel('Action')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Action Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # 3. Performance by Attack Type
            attack_types = list(self.results['attack_type_performance'].keys())
            attack_means = [np.mean(self.results['attack_type_performance'][a]) 
                           for a in attack_types]
            attack_stds = [np.std(self.results['attack_type_performance'][a]) 
                          for a in attack_types]
            
            axes[0, 2].bar(range(len(attack_types)), attack_means, 
                          yerr=attack_stds, capsize=5, color='lightcoral', edgecolor='black')
            axes[0, 2].set_xlabel('Attack Type')
            axes[0, 2].set_ylabel('Average Reward')
            axes[0, 2].set_title('Performance by Attack Type')
            axes[0, 2].set_xticks(range(len(attack_types)))
            axes[0, 2].set_xticklabels(attack_types, rotation=45, ha='right')
            axes[0, 2].grid(True, alpha=0.3, axis='y')
            
            # 4. Episode Length Distribution
            axes[1, 0].hist(self.results['episode_lengths'], bins=20, 
                           color='lightgreen', edgecolor='black')
            axes[1, 0].axvline(np.mean(self.results['episode_lengths']), 
                              color='red', linestyle='--', 
                              label=f"Mean: {np.mean(self.results['episode_lengths']):.1f}")
            axes[1, 0].set_xlabel('Episode Length (steps)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Engagement Time Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Reward over Episodes
            axes[1, 1].plot(self.results['episode_rewards'], alpha=0.5, color='blue')
            # Moving average
            window = min(10, len(self.results['episode_rewards']) // 10)
            if len(self.results['episode_rewards']) >= window:
                ma = np.convolve(self.results['episode_rewards'], 
                                np.ones(window)/window, mode='valid')
                axes[1, 1].plot(range(window-1, len(self.results['episode_rewards'])), 
                               ma, color='red', linewidth=2, label=f'MA({window})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].set_title('Rewards Over Episodes')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Action Diversity Score
            total_actions = sum(self.results['action_distribution'].values())
            action_pcts = [self.results['action_distribution'][i] / total_actions * 100 
                          for i in range(6)]
            
            # Create pie chart
            axes[1, 2].pie(action_pcts, labels=action_names, autopct='%1.1f%%',
                          colors=colors, startangle=90)
            axes[1, 2].set_title('Action Distribution (Pie Chart)')
            
            plt.tight_layout()
            
            filename = f"results/evaluation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f" Visualizations saved to: {filename}")
            plt.close()
            
        except Exception as e:
            print(f"  Could not create visualizations: {e}")
    
    def compare_with_baseline(self, baseline_results_path=None):
        """Compare current model with baseline/previous results"""
        if not baseline_results_path:
            print("  No baseline results provided for comparison")
            return
        
        try:
            with open(baseline_results_path, 'r') as f:
                baseline = json.load(f)
            
            print("\n" + "=" * 70)
            print("COMPARISON WITH BASELINE")
            print("=" * 70)
            
            current_mean = np.mean(self.results['episode_rewards'])
            baseline_mean = baseline['overall_performance']['mean_reward']
            
            improvement = ((current_mean - baseline_mean) / abs(baseline_mean)) * 100
            
            print(f"Current Model:  {current_mean:.2f}")
            print(f"Baseline Model: {baseline_mean:.2f}")
            print(f"Improvement:    {improvement:+.1f}%")
            
            if improvement > 10:
                print(" Significant improvement!")
            elif improvement > 0:
                print("✓ Marginal improvement")
            else:
                print("  Performance decreased")
            
            print("=" * 70)
            
        except Exception as e:
            print(f"  Could not load baseline results: {e}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RL Honeypot Agent')
    parser.add_argument('--model', type=str, default='models/final_agent.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline results JSON for comparison')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RL HONEYPOT AGENT - MODEL EVALUATION")
    print("=" * 70)
    print()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        num_eval_episodes=args.episodes
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Compare with baseline if provided
    if args.baseline:
        evaluator.compare_with_baseline(args.baseline)
    
    print("\n Evaluation complete!")
    print("\nGenerated files:")
    print("  - results/evaluation_YYYYMMDD_HHMMSS.json")
    print("  - results/evaluation_plots_YYYYMMDD_HHMMSS.png")


if __name__ == '__main__':
    main()
