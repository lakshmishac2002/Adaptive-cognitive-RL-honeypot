#!/usr/bin/env python3
"""
FIXED Reward Shaper for Maintaining Action Diversity
Reduced penalties to allow longer episodes
"""
import numpy as np


class DiversityRewardShaper:
    """Shapes rewards to maintain action diversity and encourage attack-specific responses"""
    
    def __init__(self, action_dim=6, diversity_bonus_weight=100.0):
        self.action_dim = action_dim
        self.diversity_bonus_weight = diversity_bonus_weight
        self.action_counts = np.zeros(action_dim)
        self.episode_actions = []
        
        # Action names for reference
        self.action_names = ['respond_realistic', 'delay_response', 'fake_error', 
                            'fake_data', 'redirect_deep', 'block']
        
        # Attack-specific bonuses (attack_type -> action_name -> bonus)
        self.attack_bonuses = {
            'port_scan': {'delay_response': 200, 'fake_data': 150, 'redirect_deep': 100},
            'sql_injection': {'fake_error': 70, 'fake_data': 60, 'delay_response': 60},
            'xss': {'fake_data': 300, 'redirect_deep': 200, 'fake_error': 150},
            'directory_traversal': {'redirect_deep': 300, 'fake_data': 200, 'delay_response': 100},
            'command_injection': {'fake_error': 250, 'fake_data': 200, 'delay_response': 150},
            'malware': {'redirect_deep': 300, 'fake_data': 250, 'delay_response': 100},
            'ssh_bruteforce': {'delay_response': 300, 'fake_error': 200, 'fake_data': 100},
            'ddos': {'delay_response': 300, 'fake_data': 200, 'redirect_deep': 150},
            'zero_day': {'fake_data': 400, 'redirect_deep': 300, 'fake_error': 200}
        }
        
        # FIXED: Less harsh penalties to allow longer episodes
        self.overuse_penalties = {
            'respond_realistic': 0.75,  # Was 0.55 (less harsh)
            'delay_response': 0.75,     # Was 0.60 (less harsh)
            'fake_error': 0.75,         # Was 0.65 (less harsh)
            'fake_data': 0.75,          # Was 0.60 (less harsh)
            'block': 0.50               # Was 0.45 (slightly less harsh)
        }
    
    def shape_reward(self, base_reward, action, attack_type, step, done):
        """
        Apply reward shaping to encourage diversity and attack-appropriate actions
        
        Args:
            base_reward: Original reward from environment
            action: Action index (0-5)
            attack_type: String attack type (e.g., 'sql_injection')
            step: Current step in episode
            done: Whether episode is finished
        
        Returns:
            Shaped reward (float)
        """
        # FIXED: Less aggressive normalization to preserve reward magnitude
        base_reward = np.tanh(base_reward / 600.0) * 400  # Was 500->300
        reward = float(base_reward)
        
        # 1. Attack-specific bonuses
        if action < len(self.action_names):
            action_name = self.action_names[action]
            if attack_type in self.attack_bonuses:
                bonus = self.attack_bonuses[attack_type].get(action_name, 0)
                reward += bonus
        
        # 2. Diversity bonus (after initial exploration)
        total = self.action_counts.sum()
        if total > 100:
            freq = self.action_counts[action] / total
            expected = 1.0 / self.action_dim
            
            # Bonus for underused actions
            if freq < expected:
                diversity_bonus = self.diversity_bonus_weight * 3.0 * (expected - freq)
                reward += diversity_bonus
            
            # FIXED: Apply penalty only if significantly overused
            if freq > expected * 1.5:  # Was 1.3 (more lenient)
                action_name = self.action_names[action]
                if action_name in self.overuse_penalties:
                    reward *= self.overuse_penalties[action_name]
        
        # 3. Episode diversity bonus
        self.episode_actions.append(action)
        if done:
            unique = len(set(self.episode_actions))
            reward += unique * 120  # Bonus for using diverse actions
            self.episode_actions = []
        
        # 4. FIXED: Enhanced engagement bonus
        reward += step * 3  # Was 2 (more reward for longer episodes)
        
        # 5. FIXED: Penalty for blocking already handled in environment
        # Removed redundant penalty here
        
        # Update counts
        self.action_counts[action] += 1
        
        return reward
    
    def get_statistics(self):
        """Get diversity statistics"""
        total = self.action_counts.sum()
        if total == 0:
            return {'diversity_percent': 0}
        
        freqs = self.action_counts / total
        entropy = -np.sum(freqs * np.log(freqs + 1e-10))
        max_entropy = np.log(self.action_dim)
        
        return {
            'action_counts': self.action_counts.tolist(),
            'action_frequencies': freqs.tolist(),
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'diversity_percent': float(entropy / max_entropy * 100)
        }


# Quick test
if __name__ == "__main__":
    print("Testing FIXED Reward Shaper")
    print("=" * 60)
    
    shaper = DiversityRewardShaper()
    
    print("\n✓ FIXES APPLIED:")
    print("  • Overuse penalties: 0.55-0.65 → 0.75 (less harsh)")
    print("  • Overuse threshold: 1.3x → 1.5x (more lenient)")
    print("  • Engagement multiplier: 2 → 3 (stronger)")
    print("  • Normalization: 500→300 → 600→400 (preserve magnitude)")
    print()
    
    # Test scenario: balanced usage
    print("\nScenario 1: Balanced usage")
    shaper.action_counts = np.array([100, 100, 100, 100, 100, 100])
    base = 50.0
    shaped = shaper.shape_reward(base, 0, 'sql_injection', 10, False)
    print(f"  Base: {base:.2f}, Shaped: {shaped:.2f}")
    stats = shaper.get_statistics()
    print(f"  Diversity: {stats['diversity_percent']:.1f}%")
    
    # Test scenario: moderate overuse (should be more lenient now)
    print("\nScenario 2: Moderate overuse (more lenient with fixes)")
    shaper.action_counts = np.array([400, 100, 100, 100, 100, 100])
    shaped_over = shaper.shape_reward(base, 0, 'sql_injection', 10, False)
    shaped_under = shaper.shape_reward(base, 2, 'sql_injection', 10, False)
    print(f"  Base: {base:.2f}")
    print(f"  Shaped (overused action): {shaped_over:.2f} [less harsh penalty]")
    print(f"  Shaped (underused action): {shaped_under:.2f} [bonus applied]")
    stats = shaper.get_statistics()
    print(f"  Diversity: {stats['diversity_percent']:.1f}%")
    
    print("\n✓ Reward shaper should now allow better exploration!")
    print("=" * 60)
