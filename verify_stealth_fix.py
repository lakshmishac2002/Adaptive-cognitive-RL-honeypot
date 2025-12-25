#!/usr/bin/env python3
"""
Quick verification script to test if stealth fix is working
"""
import torch
import numpy as np
from honeypot_environment_enhanced import EnhancedHoneypotEnvironment
from custom_rl_agent import FixedHoneypotRLAgent


def quick_stealth_test(model_path='models/final_agent.pth', num_tests=20):
    """Quick test to verify stealth improvements"""
    
    print("=" * 80)
    print("STEALTH FIX VERIFICATION TEST")
    print("=" * 80)
    print()
    
    # Load environment and agent
    env = EnhancedHoneypotEnvironment()
    agent = FixedHoneypotRLAgent(env.state_dim, env.action_dim)
    
    try:
        checkpoint = torch.load(model_path, map_location=agent.device, weights_only=False)
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.epsilon = 0.0
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"⚠ Could not load model: {e}")
        print("  Using random actions for testing...")
        agent = None
    
    print()
    print(f"Testing {num_tests} episodes with stealth-fixed environment...")
    print("-" * 80)
    
    # Metrics tracking
    stealth_maintained = 0
    episode_lengths = []
    final_suspicions = []
    intelligence_gathered = []
    total_rewards = []
    
    for episode in range(num_tests):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            if agent:
                action = agent.select_action(state, training=False)
            else:
                # Random action (avoid blocking)
                action = np.random.randint(0, env.action_dim - 1)
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
        
        # Record metrics
        episode_lengths.append(steps)
        final_suspicions.append(info['suspicious_level'])
        intelligence_gathered.append(info['intelligence'])
        total_rewards.append(total_reward)
        
        # Check stealth
        if info['suspicious_level'] < 0.75:
            stealth_maintained += 1
            status = "✓ STEALTH OK"
        else:
            status = "✗ DETECTED"
        
        if (episode + 1) % 5 == 0:
            print(f"Episodes {episode-3:2d}-{episode+1:2d}: "
                  f"Avg Steps={np.mean(episode_lengths[-5:]):5.1f}, "
                  f"Avg Suspicion={np.mean(final_suspicions[-5:]):.3f}, "
                  f"Stealth Rate={sum([1 for s in final_suspicions[-5:] if s < 0.75])/5*100:.0f}%")
    
    print()
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Calculate statistics
    stealth_rate = (stealth_maintained / num_tests) * 100
    avg_length = np.mean(episode_lengths)
    avg_suspicion = np.mean(final_suspicions)
    avg_intel = np.mean(intelligence_gathered)
    avg_reward = np.mean(total_rewards)
    
    print(f"Episode Length:")
    print(f"  Average: {avg_length:.1f} steps")
    print(f"  Min/Max: {np.min(episode_lengths)}/{np.max(episode_lengths)} steps")
    print()
    
    print(f"Suspicion Levels:")
    print(f"  Average Final: {avg_suspicion:.3f}")
    print(f"  Min/Max: {np.min(final_suspicions):.3f}/{np.max(final_suspicions):.3f}")
    print()
    
    print(f"Stealth Performance:")
    print(f"  Episodes < 0.75 suspicion: {stealth_maintained}/{num_tests}")
    print(f"  Stealth Success Rate: {stealth_rate:.1f}%")
    print()
    
    print(f"Intelligence Gathered:")
    print(f"  Average: {avg_intel:.1f} units/episode")
    print()
    
    print(f"Reward Performance:")
    print(f"  Average: {avg_reward:.1f} per episode")
    print()
    
    # Evaluation
    print("=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print()
    
    all_good = True
    
    # Check 1: Episode Length
    if avg_length >= 45:
        print("✅ Episode Length: GOOD (>45 steps)")
    elif avg_length >= 35:
        print("⚠️ Episode Length: ACCEPTABLE (35-45 steps)")
        all_good = False
    else:
        print("❌ Episode Length: LOW (<35 steps)")
        all_good = False
    
    # Check 2: Stealth Rate
    if stealth_rate >= 60:
        print(f"✅ Stealth Rate: EXCELLENT ({stealth_rate:.1f}% ≥ 60%)")
    elif stealth_rate >= 40:
        print(f"⚠️ Stealth Rate: ACCEPTABLE ({stealth_rate:.1f}% - needs improvement)")
        all_good = False
    else:
        print(f"❌ Stealth Rate: LOW ({stealth_rate:.1f}% < 40%)")
        all_good = False
    
    # Check 3: Intelligence
    if avg_intel >= 70:
        print(f"✅ Intelligence: EXCELLENT ({avg_intel:.1f} ≥ 70)")
    elif avg_intel >= 50:
        print(f"⚠️ Intelligence: ACCEPTABLE ({avg_intel:.1f} units)")
        all_good = False
    else:
        print(f"❌ Intelligence: LOW ({avg_intel:.1f} < 50)")
        all_good = False
    
    # Check 4: Average Suspicion
    if avg_suspicion < 0.65:
        print(f"✅ Avg Suspicion: EXCELLENT ({avg_suspicion:.3f} < 0.65)")
    elif avg_suspicion < 0.72:
        print(f"⚠️ Avg Suspicion: ACCEPTABLE ({avg_suspicion:.3f})")
        all_good = False
    else:
        print(f"❌ Avg Suspicion: HIGH ({avg_suspicion:.3f} ≥ 0.72)")
        all_good = False
    
    print()
    print("=" * 80)
    
    if all_good:
        print("🎉 STEALTH FIX SUCCESSFUL!")
        print()
        print("All metrics are excellent. You can now proceed with:")
        print("  1. Full evaluation: python3 evaluate_model.py --model models/final_agent.pth")
        print("  2. Cumulative scoring: python3 calculate_cumulative_scores.py --model models/final_agent.pth --episodes 200")
    elif stealth_rate >= 40:
        print("✓ PARTIAL SUCCESS - Stealth Improved!")
        print()
        print("Stealth rate improved but could be better. Consider:")
        print("  1. Reducing suspicion rates by another 10-20%")
        print("  2. Increasing stealth maintenance bonuses to +20/+12")
        print("  3. Run full evaluation to see comprehensive results")
    else:
        print("⚠️ STEALTH FIX NEEDS ADJUSTMENT")
        print()
        print("Recommendations:")
        print("  1. Reduce suspicion rates by additional 30%:")
        print("     'respond_realistic': 0.0024 → 0.0017")
        print("     'delay_response': 0.0048 → 0.0034")
        print("     'fake_error': 0.0072 → 0.0050")
        print("  2. Increase detection threshold: 0.75 → 0.80")
        print("  3. Boost stealth bonuses: +15 → +25")
    
    print("=" * 80)
    print()
    
    return {
        'stealth_rate': stealth_rate,
        'avg_length': avg_length,
        'avg_suspicion': avg_suspicion,
        'avg_intel': avg_intel,
        'avg_reward': avg_reward
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/final_agent.pth')
    parser.add_argument('--tests', type=int, default=20)
    
    args = parser.parse_args()
    
    results = quick_stealth_test(args.model, args.tests)
