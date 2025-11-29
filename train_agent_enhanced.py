#!/usr/bin/env python3
"""
Enhanced training script with better monitoring and diagnosis
"""
import sys
import json
import numpy as np
from datetime import datetime
from reward_shaper import DiversityRewardShaper
from diversity_enforcing_wrapper import wrap_environment
from custom_rl_agent import FixedHoneypotRLAgent  # Make sure this matches your file name
from honeypot_environment_enhanced import EnhancedHoneypotEnvironment


def print_action_distribution(agent, action_names):
    """Print detailed action distribution"""
    stats = agent.get_action_statistics()
    
    print("\n" + "=" * 70)
    print("ACTION USAGE STATISTICS")
    print("=" * 70)
    
    if 'action_counts' in stats:
        counts = stats['action_counts']
        percentages = stats['action_percentages']
        avg_rewards = stats['action_avg_rewards']
        
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            action_name = action_names.get(i, f"Action {i}")
            avg_reward = avg_rewards.get(i, 0.0)
            print(f"{action_name:25s}: {int(count):6d} times ({pct:5.1f}%) | "
                  f"Avg Reward: {avg_reward:8.2f}")


def train(episodes=1000, save_freq=100, verbose=True):
    """Enhanced training with better monitoring"""
    
    print("=" * 70)
    print("RL HONEYPOT AGENT - ENHANCED TRAINING")
    print("=" * 70)
    print()
    
    # Initialize environment
    env = EnhancedHoneypotEnvironment()
    env = wrap_environment(env, diversity_weight=40)
    
    print("Attack Types Covered:")
    for idx, attack in env.attack_types.items():
        print(f"  {idx+1}. {attack}")
    print()
    
    print("Response Actions:")
    action_names = {}
    for idx, action in env.actions.items():
        print(f"  {idx}. {action}")
        action_names[idx] = action
    print()
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Features:")
    print(f"  ✓ Dueling DQN architecture")
    print(f"  ✓ Prioritized Experience Replay")
    print(f"  ✓ Enhanced exploration with action balancing")
    print(f"  ✓ Exploration bonus for underused actions")
    print(f"  ✓ Slower epsilon decay (better exploration)")
    print(f"  ✓ Entropy regularization")
    print(f"  ✓ Intrinsic motivation bonuses")
    print("=" * 70)
    print()
    
    # Initialize agent
    agent = FixedHoneypotRLAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=0.0003,
        epsilon_end=0.15,
        entropy_coef=0.02,
        diversity_bonus_weight=50.0
    )
    
    reward_shaper = DiversityRewardShaper(
        action_dim=env.action_dim,
        diversity_bonus_weight=100.0  # Reduced from 150
    )
    
    # Tracking
    episode_rewards = []
    episode_raw_rewards = []  # Track raw rewards separately
    episode_losses = []
    attack_type_performance = {attack: [] for attack in env.attack_types.values()}
    best_reward = -float('inf')
    best_diversity = 0
    
    # Action diversity tracking
    action_diversity_history = []
    top_action_usage_history = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_raw_reward = 0  # Track raw reward
        episode_loss = []
        done = False
        attack_type = env.attack_types[env.current_attack_type]
        step = 0
        
        agent.reset_episode_tracking()  # Reset episode-level tracking
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, raw_reward, done, info = env.step(action)
            
            # Apply reward shaping
            shaped_reward = reward_shaper.shape_reward(
                base_reward=raw_reward,
                action=action,
                attack_type=attack_type,
                step=step,
                done=done
            )
            
            #  CORRECT PLACEMENT: Compute intrinsic motivation reward
            intrinsic_reward = agent.compute_intrinsic_reward(state, action)
            
            #  Combine shaped + intrinsic rewards
            total_reward = shaped_reward + intrinsic_reward
            
            #  Store transition with total reward
            agent.store_transition(state, action, total_reward, next_state, done)
            
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            episode_reward += total_reward
            episode_raw_reward += raw_reward
            state = next_state
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_raw_rewards.append(episode_raw_reward)
        attack_type_performance[attack_type].append(episode_reward)
        
        if episode_loss:
            episode_losses.append(np.mean(episode_loss))
        
        # Track action diversity
        stats = agent.get_action_statistics()
        if 'action_percentages' in stats:
            # Calculate entropy as diversity metric
            probs = np.array(stats['action_percentages']) / 100.0
            probs = probs[probs > 0]  # Remove zeros
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            action_diversity_history.append(entropy)
            
            # Track top action usage
            percentages = stats['action_percentages']
            top_action_usage_history.append(max(percentages))
        
        # Periodic logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_raw = np.mean(episode_raw_rewards[-10:])
            avg_loss = np.mean(episode_losses[-10:]) if episode_losses else 0
            shaper_stats = reward_shaper.get_statistics()
            diversity = shaper_stats.get('diversity_percent', 0)
            current_top_usage = top_action_usage_history[-1] if top_action_usage_history else 0
            
            if verbose:
                print(f"Episode {episode + 1:4d}/{episodes} | "
                      f"Attack: {attack_type:20s} | "
                      f"Shaped: {episode_reward:7.2f} | "
                      f"Raw: {episode_raw_reward:7.2f} | "
                      f"Avg(10): {avg_reward:7.2f} | "
                      f"Div: {diversity:5.1f}% | "
                      f"Top: {current_top_usage:4.1f}% | "
                      f"Loss: {avg_loss:6.4f} | "
                      f"ε: {agent.epsilon:.4f} | "
                      f"Steps: {step:3d}")
        
        # Detailed checkpoint logging
        if (episode + 1) % save_freq == 0:
            print()
            print("=" * 70)
            print(f"CHECKPOINT - Episode {episode + 1}")
            print("=" * 70)
            
            # Save model
            agent.save(f'models/agent_episode_{episode + 1}.pth')
            
            # Performance metrics
            avg_reward_100 = np.mean(episode_rewards[-100:])
            avg_raw_100 = np.mean(episode_raw_rewards[-100:])
            std_reward_100 = np.std(episode_rewards[-100:])
            
            print(f"\nPerformance (last 100 episodes):")
            print(f"  Shaped Reward: {avg_reward_100:.2f} ± {std_reward_100:.2f}")
            print(f"  Raw Reward: {avg_raw_100:.2f}")
            print(f"  Min/Max: {np.min(episode_rewards[-100:]):.2f} / {np.max(episode_rewards[-100:]):.2f}")
            
            if avg_reward_100 > best_reward:
                best_reward = avg_reward_100
                agent.save('models/best_agent.pth')
                print(f"\n   NEW BEST MODEL! Avg reward: {avg_reward_100:.2f}")
            
            # Action distribution
            print_action_distribution(agent, action_names)
            
            # Action diversity metric
            if action_diversity_history:
                recent_diversity = np.mean(action_diversity_history[-100:])
                max_entropy = np.log(env.action_dim)  # Maximum possible entropy
                diversity_pct = (recent_diversity / max_entropy) * 100
                recent_top_usage = np.mean(top_action_usage_history[-100:]) if len(top_action_usage_history) >= 100 else 0
                
                print(f"\nAction Diversity: {recent_diversity:.3f} / {max_entropy:.3f} "
                      f"({diversity_pct:.1f}% of maximum)")
                print(f"Top Action Usage: {recent_top_usage:.1f}%")
                
                if diversity_pct < 30:
                    print("    WARNING: Low action diversity! Agent may be stuck.")
                elif diversity_pct > 70:
                    print("  ✓ Good action diversity!")
                
                # Save based on diversity
                if diversity_pct > best_diversity:
                    best_diversity = diversity_pct
                    agent.save('models/best_diversity_agent.pth')
                    print(f"   NEW BEST DIVERSITY! {diversity_pct:.1f}%")
            
            print("=" * 70)
            print()
    
    # Save final model
    agent.save('models/final_agent.pth')
    
    # Final statistics
    print()
    print("=" * 70)
    print("FINAL PERFORMANCE BY ATTACK TYPE")
    print("=" * 70)
    
    attack_stats = {}
    for attack_type, rewards in attack_type_performance.items():
        if rewards:
            avg = np.mean(rewards)
            std = np.std(rewards)
            attack_stats[attack_type] = {'avg': avg, 'std': std, 'count': len(rewards)}
            print(f"{attack_type:20s}: {avg:8.2f} ± {std:6.2f} ({len(rewards)} episodes)")
    
    # Final action distribution
    print_action_distribution(agent, action_names)
    
    # Final diversity report
    final_diversity = action_diversity_history[-1] if action_diversity_history else 0
    final_top_usage = top_action_usage_history[-1] if top_action_usage_history else 0
    max_entropy = np.log(env.action_dim)
    final_diversity_pct = (final_diversity / max_entropy) * 100
    
    print(f"\nFinal Diversity Metrics:")
    print(f"  Diversity: {final_diversity_pct:.1f}%")
    print(f"  Top Action Usage: {final_top_usage:.1f}%")
    print(f"  Best Diversity Achieved: {best_diversity:.1f}%")
    
    # Save comprehensive results
    results = {
        'episode_rewards': episode_rewards,
        'episode_raw_rewards': episode_raw_rewards,
        'episode_losses': episode_losses,
        'action_diversity_history': action_diversity_history,
        'top_action_usage_history': top_action_usage_history,
        'best_reward': float(best_reward),
        'best_diversity': float(best_diversity),
        'attack_type_performance': {
            k: {'avg': float(v['avg']), 'std': float(v['std']), 'count': v['count']}
            for k, v in attack_stats.items()
        },
        'final_action_stats': agent.get_action_statistics(),
        'total_episodes': episodes,
        'final_epsilon': float(agent.epsilon),
        'training_steps': agent.training_step
    }
    
    with open('results/training_history_enhanced.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print(f"Best average reward: {best_reward:.2f}")
    print(f"Best diversity: {best_diversity:.1f}%")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Results saved to: results/training_history_enhanced.json")
    print("=" * 70)
    
    return agent, results


def diagnose_agent(filepath='models/best_agent.pth'):
    """Diagnose a trained agent's behavior"""
    
    print("\n" + "=" * 70)
    print("AGENT DIAGNOSIS")
    print("=" * 70)
    
    env = EnhancedHoneypotEnvironment()
    agent = FixedHoneypotRLAgent(env.state_dim, env.action_dim)
    
    # Fix for PyTorch 2.6+ compatibility
    import torch
    checkpoint = torch.load(filepath, map_location=agent.device, weights_only=False)
    agent.policy_net.load_state_dict(checkpoint['policy_net'])
    agent.target_net.load_state_dict(checkpoint['target_net'])
    agent.training_step = checkpoint['training_step']
    agent.epsilon = checkpoint['epsilon']
    print(f" Model loaded from {filepath}")
    print(f"   Training step: {agent.training_step}")
    print(f"   Epsilon: {agent.epsilon:.4f}")
    
    # Set to evaluation mode
    agent.epsilon = 0.15
    
    print("\nTesting agent on 50 random states...")
    
    # Sample random states and see what agent does
    action_counts = np.zeros(env.action_dim)
    
    for _ in range(50):
        state = env.reset()
        action = agent.select_action(state, training=False)
        action_counts[action] += 1
    
    print("\nAction distribution on random states:")
    for i, count in enumerate(action_counts):
        action_name = env.actions[i]
        print(f"  {action_name:25s}: {int(count):3d} times ({count/50*100:5.1f}%)")
    
    # Check Q-values for a sample state
    print("\nSample Q-values for a random state:")
    state = env.reset()
    
    import torch
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
    
    for i, q_val in enumerate(q_values):
        action_name = env.actions[i]
        print(f"  {action_name:25s}: Q = {q_val:8.3f}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL Honeypot Agent')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--save-freq', type=int, default=100, help='Save frequency')
    parser.add_argument('--diagnose', type=str, default=None, help='Diagnose a saved model')
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_agent(args.diagnose)
    else:
        agent, results = train(episodes=args.episodes, save_freq=args.save_freq)
        
        # Automatically diagnose the final agent
        print("\n\nDiagnosing final agent...")
        diagnose_agent('models/best_agent.pth')
