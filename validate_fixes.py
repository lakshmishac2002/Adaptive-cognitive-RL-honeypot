#!/usr/bin/env python3
"""
Comprehensive Validation Script
Tests that all fixes are working correctly before full training
"""
import numpy as np
import sys
from honeypot_environment_enhanced import FixedHoneypotEnvironment


def test_stealth_mechanics():
    """Test that stealth mechanics are fixed"""
    print("\n" + "=" * 70)
    print("TEST 1: STEALTH MECHANICS")
    print("=" * 70)
    
    env = FixedHoneypotEnvironment()
    
    # Test each attack type with good actions
    stealth_successes = 0
    total_tests = 0
    
    attack_results = []
    
    for attack_idx in range(9):
        env.current_attack_type = attack_idx
        state = env.reset()
        attack_name = env.attack_types[attack_idx]
        
        # Use optimal actions for this attack
        optimal_map = {
            'port_scan': 1,          # delay_response
            'ssh_bruteforce': 3,     # fake_data
            'sql_injection': 2,      # fake_error
            'xss': 3,                # fake_data
            'ddos': 1,               # delay_response (or block)
            'malware': 4,            # redirect_deep
            'directory_traversal': 4, # redirect_deep
            'command_injection': 2,   # fake_error
            'zero_day': 0            # respond_realistic
        }
        
        action = optimal_map[attack_name]
        
        # Run 30 steps with optimal action
        for _ in range(30):
            state, reward, done, info = env.step(action)
            if done:
                break
        
        suspicious_level = info['suspicious_level']
        stealth_success = suspicious_level < 0.7
        
        if stealth_success:
            stealth_successes += 1
        total_tests += 1
        
        status = "✓ PASS" if stealth_success else "✗ FAIL"
        attack_results.append({
            'attack': attack_name,
            'suspicion': suspicious_level,
            'success': stealth_success
        })
        
        print(f"  {attack_name:20s}: Suspicion={suspicious_level:.3f} | {status}")
    
    stealth_rate = (stealth_successes / total_tests) * 100
    
    print(f"\nStealth Success Rate: {stealth_rate:.1f}% ({stealth_successes}/{total_tests})")
    
    if stealth_rate >= 70:
        print("✅ STEALTH MECHANICS: FIXED!")
        return True, attack_results
    elif stealth_rate >= 50:
        print("⚠️  STEALTH MECHANICS: IMPROVED (but could be better)")
        return True, attack_results
    else:
        print("❌ STEALTH MECHANICS: STILL BROKEN")
        return False, attack_results


def test_positive_rewards():
    """Test that base rewards are positive for good actions"""
    print("\n" + "=" * 70)
    print("TEST 2: POSITIVE BASE REWARDS")
    print("=" * 70)
    
    env = FixedHoneypotEnvironment()
    
    positive_reward_count = 0
    total_episodes = 0
    
    reward_results = []
    
    for attack_idx in range(9):
        env.current_attack_type = attack_idx
        state = env.reset()
        attack_name = env.attack_types[attack_idx]
        
        # Use optimal actions
        optimal_map = {
            'port_scan': 1, 'ssh_bruteforce': 3, 'sql_injection': 2,
            'xss': 3, 'ddos': 1, 'malware': 4,
            'directory_traversal': 4, 'command_injection': 2, 'zero_day': 0
        }
        
        action = optimal_map[attack_name]
        
        episode_reward = 0
        for _ in range(20):
            state, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        
        if episode_reward > 0:
            positive_reward_count += 1
        total_episodes += 1
        
        status = "✓ PASS" if episode_reward > 0 else "✗ FAIL"
        reward_results.append({
            'attack': attack_name,
            'reward': episode_reward,
            'positive': episode_reward > 0
        })
        
        print(f"  {attack_name:20s}: Reward={episode_reward:7.1f} | {status}")
    
    positive_rate = (positive_reward_count / total_episodes) * 100
    
    print(f"\nPositive Reward Rate: {positive_rate:.1f}% ({positive_reward_count}/{total_episodes})")
    
    if positive_rate >= 80:
        print("✅ POSITIVE REWARDS: FIXED!")
        return True, reward_results
    elif positive_rate >= 60:
        print("⚠️  POSITIVE REWARDS: IMPROVED")
        return True, reward_results
    else:
        print("❌ POSITIVE REWARDS: STILL BROKEN")
        return False, reward_results


def test_sophisticated_attacks():
    """Test handling of zero-day, XSS, and malware"""
    print("\n" + "=" * 70)
    print("TEST 3: SOPHISTICATED ATTACK HANDLING")
    print("=" * 70)
    
    env = FixedHoneypotEnvironment()
    
    sophisticated_attacks = {
        'zero_day': 0,       # respond_realistic
        'xss': 3,            # fake_data
        'malware': 4         # redirect_deep
    }
    
    results = {}
    all_passed = True
    
    for attack_name, optimal_action in sophisticated_attacks.items():
        # Find attack index
        attack_idx = None
        for idx, name in env.attack_types.items():
            if name == attack_name:
                attack_idx = idx
                break
        
        env.current_attack_type = attack_idx
        state = env.reset()
        
        episode_reward = 0
        for _ in range(30):
            state, reward, done, info = env.step(optimal_action)
            episode_reward += reward
            if done:
                break
        
        suspicious_level = info['suspicious_level']
        intel = info['intelligence']
        engagement = info['engaged_time']
        
        # Success criteria
        good_reward = episode_reward > 500
        good_stealth = suspicious_level < 0.7
        good_intel = intel > 10
        
        success = good_reward and (good_stealth or good_intel)
        
        results[attack_name] = {
            'reward': episode_reward,
            'stealth': suspicious_level,
            'intel': intel,
            'engagement': engagement,
            'success': success
        }
        
        status = "✓ PASS" if success else "✗ FAIL"
        if not success:
            all_passed = False
        
        print(f"\n  {attack_name.upper()}:")
        print(f"    Reward: {episode_reward:.1f} | {'✓' if good_reward else '✗'}")
        print(f"    Stealth: {suspicious_level:.3f} | {'✓' if good_stealth else '✗'}")
        print(f"    Intelligence: {intel} | {'✓' if good_intel else '✗'}")
        print(f"    Status: {status}")
    
    if all_passed:
        print("\n✅ SOPHISTICATED ATTACKS: FIXED!")
        return True, results
    else:
        print("\n⚠️  SOPHISTICATED ATTACKS: PARTIALLY FIXED")
        return False, results


def test_action_effectiveness():
    """Test that different actions have different effectiveness"""
    print("\n" + "=" * 70)
    print("TEST 4: ACTION-SPECIFIC EFFECTIVENESS")
    print("=" * 70)
    
    env = FixedHoneypotEnvironment()
    
    # Test SQL injection with different actions
    env.current_attack_type = 2  # SQL injection
    
    action_rewards = {}
    
    for action_idx in range(6):
        state = env.reset()
        action_name = env.actions[action_idx]
        
        episode_reward = 0
        for _ in range(20):
            state, reward, done, info = env.step(action_idx)
            episode_reward += reward
            if done:
                break
        
        action_rewards[action_name] = episode_reward
        print(f"  {action_name:20s}: {episode_reward:7.1f}")
    
    # Check if optimal actions get better rewards
    sorted_actions = sorted(action_rewards.items(), key=lambda x: x[1], reverse=True)
    best_action = sorted_actions[0][0]
    
    # For SQL injection, fake_error or fake_data should be best
    expected_best = ['fake_error', 'fake_data']
    
    if best_action in expected_best:
        print(f"\n✅ ACTION EFFECTIVENESS: FIXED! (Best: {best_action})")
        return True, action_rewards
    else:
        print(f"\n⚠️  ACTION EFFECTIVENESS: Best is {best_action}, expected {expected_best}")
        return False, action_rewards


def test_complete_episode():
    """Run a complete episode and check all metrics"""
    print("\n" + "=" * 70)
    print("TEST 5: COMPLETE EPISODE SIMULATION")
    print("=" * 70)
    
    env = FixedHoneypotEnvironment()
    state = env.reset()
    
    episode_reward = 0
    actions_taken = []
    
    print(f"\n  Attack Type: {env.attack_types[env.current_attack_type]}")
    print(f"  Running episode with mixed actions...\n")
    
    for step in range(50):
        # Use reasonable actions (not just random)
        action = np.random.choice([1, 2, 3, 4], p=[0.3, 0.2, 0.3, 0.2])
        actions_taken.append(action)
        
        state, reward, done, info = env.step(action)
        episode_reward += reward
        
        if step < 5:
            print(f"  Step {step+1}: Action={env.actions[action]:20s} | "
                  f"Reward={reward:6.1f} | Suspicion={info['suspicious_level']:.3f}")
        
        if done:
            break
    
    print(f"\n  Episode Results:")
    print(f"    Total Reward: {episode_reward:.1f}")
    print(f"    Steps: {info['step']}")
    print(f"    Intelligence: {info['intelligence']}")
    print(f"    Engagement: {info['engaged_time']}")
    print(f"    Suspicion: {info['suspicious_level']:.3f}")
    
    # Success criteria
    success = (
        episode_reward > 200 and
        info['intelligence'] > 5 and
        info['suspicious_level'] < 0.9
    )
    
    if success:
        print("\n✅ COMPLETE EPISODE: SUCCESS!")
        return True
    else:
        print("\n⚠️  COMPLETE EPISODE: Needs improvement")
        return False


def run_all_tests():
    """Run all validation tests"""
    print("\n")
    print("=" * 70)
    print("COMPREHENSIVE FIX VALIDATION")
    print("=" * 70)
    print("\nTesting all fixes before full training...\n")
    
    results = {}
    
    # Test 1: Stealth
    stealth_passed, stealth_data = test_stealth_mechanics()
    results['stealth'] = {'passed': stealth_passed, 'data': stealth_data}
    
    # Test 2: Positive rewards
    rewards_passed, reward_data = test_positive_rewards()
    results['rewards'] = {'passed': rewards_passed, 'data': reward_data}
    
    # Test 3: Sophisticated attacks
    sophisticated_passed, sophisticated_data = test_sophisticated_attacks()
    results['sophisticated'] = {'passed': sophisticated_passed, 'data': sophisticated_data}
    
    # Test 4: Action effectiveness
    actions_passed, action_data = test_action_effectiveness()
    results['actions'] = {'passed': actions_passed, 'data': action_data}
    
    # Test 5: Complete episode
    episode_passed = test_complete_episode()
    results['episode'] = {'passed': episode_passed}
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    tests = [
        ("Stealth Mechanics", stealth_passed),
        ("Positive Base Rewards", rewards_passed),
        ("Sophisticated Attacks", sophisticated_passed),
        ("Action Effectiveness", actions_passed),
        ("Complete Episode", episode_passed)
    ]
    
    passed_count = sum(1 for _, passed in tests if passed)
    total_count = len(tests)
    
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:25s}: {status}")
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL FIXES VALIDATED! Ready for full training!")
        return True
    elif passed_count >= total_count - 1:
        print("\n✅ Most fixes validated! Safe to proceed with training.")
        return True
    else:
        print("\n⚠️  Some issues remain. Review failed tests before training.")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("HONEYPOT ENVIRONMENT FIX VALIDATOR")
    print("=" * 70)
    print("\nThis script validates all fixes:")
    print("  1. Stealth mechanics (reduced suspicion)")
    print("  2. Positive base rewards")
    print("  3. Sophisticated attack handling (zero-day, XSS, malware)")
    print("  4. Action-specific effectiveness")
    print("  5. Complete episode simulation")
    print("\n" + "=" * 70)
    
    input("\nPress Enter to start validation...")
    
    success = run_all_tests()
    
    if success:
        print("\n" + "=" * 70)
        print("READY TO TRAIN!")
        print("=" * 70)
        print("\nRun the training script:")
        print("  python train_agent_optimized.py --episodes 500")
        print("\nExpected improvements:")
        print("  • Stealth success rate: 50-80% (was 0%)")
        print("  • Positive raw rewards in most episodes")
        print("  • 80%+ success on zero-day, XSS, malware")
        print("  • Maintained action diversity (70%+)")
        print("  • Overall success rate: 75-85%")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("FIX ISSUES BEFORE TRAINING")
        print("=" * 70)
        print("\nSome fixes need adjustment. Review test output above.")
    
    sys.exit(0 if success else 1)
