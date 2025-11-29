#!/usr/bin/env python3
"""
Environment Wrapper with Diversity Enforcement
This wraps your existing environment to force action diversity
"""
import numpy as np
from collections import deque


class DiversityEnforcingWrapper:
    """
    Wraps honeypot environment to enforce action diversity
    Uses a combination of:
    1. Action history tracking
    2. Strong penalties for repetitive behavior
    3. Bonus rewards for diverse action sequences
    """
    
    def __init__(self, env, window_size=20, diversity_weight=40.0, hard_cap=0.25):  #diversity_wight=300.0
        """
        Args:
            env: Base honeypot environment
            window_size: Look back this many actions
            diversity_weight: Strength of diversity enforcement
            hard_cap: Maximum allowed frequency for any action (0.25 = 25%)
        """
        self.env = env
        self.window_size = window_size
        self.diversity_weight = diversity_weight
        self.hard_cap = hard_cap
        
        # Track recent actions
        self.action_history = deque(maxlen=window_size)
        self.global_action_counts = np.zeros(env.action_dim)
        self.episode_action_counts = np.zeros(env.action_dim)
        
        # Pass through attributes
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.attack_types = env.attack_types
        self.actions = env.actions
        self.current_attack_type = 0
        
    def reset(self):
        """Reset environment and diversity tracking"""
        self.episode_action_counts = np.zeros(self.action_dim)
        state = self.env.reset()
        self.current_attack_type = self.env.current_attack_type
        return state
    
    def step(self, action):
        """Step with diversity-aware reward modification"""
        # Take action in base environment
        next_state, base_reward, done, info = self.env.step(action)
        
        # Track actions
        self.action_history.append(action)
        self.global_action_counts[action] += 1
        self.episode_action_counts[action] += 1
        
        # Calculate diversity-modified reward
        reward = self._modify_reward(base_reward, action, done)
        
        return next_state, reward, done, info
    
    def _modify_reward(self, base_reward, action, done):
        """Apply strong diversity enforcement"""
        reward = float(base_reward)
        
        # 1. Penalize if this action is overused globally
        total_global = self.global_action_counts.sum()
        if total_global > 100:
            global_freq = self.global_action_counts[action] / total_global
            expected_freq = 1.0 / self.action_dim
            
            # Strong penalty for overused actions
            if global_freq > expected_freq * 1.2:  # Over 20%
                overuse_factor = global_freq / expected_freq
                penalty = self.diversity_weight * (overuse_factor - 1.2)
                reward -= penalty
                
                # Extra harsh if really overused
                if global_freq > expected_freq * 2.0:  # Over 33%
                    reward *= 0.5  # Cut reward in half
        
        # 2. Penalize repetitive recent actions
        if len(self.action_history) >= 5:
            recent_5 = list(self.action_history)[-5:]
            action_count = recent_5.count(action)
            
            # Penalty grows exponentially with repetition
            if action_count >= 3:
                reward -= self.diversity_weight * (action_count - 2)
            
            # Bonus for using new action
            if action_count == 1:
                reward += self.diversity_weight * 0.5
        
        # 3. Bonus for balanced episode-level distribution
        total_episode = self.episode_action_counts.sum()
        if done and total_episode > 0:
            episode_freqs = self.episode_action_counts / total_episode
            
            # Calculate entropy (measure of diversity)
            nonzero_freqs = episode_freqs[episode_freqs > 0]
            if len(nonzero_freqs) > 0:
                entropy = -np.sum(nonzero_freqs * np.log(nonzero_freqs + 1e-10))
                max_entropy = np.log(self.action_dim)
                
                # Big bonus for high entropy (diverse actions)
                diversity_ratio = entropy / max_entropy
                reward += self.diversity_weight * diversity_ratio * 2.0
        
        # 4. Reward for using underutilized actions
        if total_global > 100:
            global_freq = self.global_action_counts[action] / total_global
            expected_freq = 1.0 / self.action_dim
            
            if global_freq < expected_freq * 0.8:  # Under 13%
                underuse_bonus = self.diversity_weight * (expected_freq * 0.8 - global_freq)
                reward += underuse_bonus
        
        return reward
    
    def get_diversity_stats(self):
        """Get current diversity statistics"""
        total = self.global_action_counts.sum()
        if total == 0:
            return None
        
        freqs = self.global_action_counts / total
        entropy = -np.sum(freqs[freqs > 0] * np.log(freqs[freqs > 0] + 1e-10))
        max_entropy = np.log(self.action_dim)
        
        return {
            'action_counts': self.global_action_counts.tolist(),
            'action_frequencies': freqs.tolist(),
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'diversity_percent': float(entropy / max_entropy * 100)
        }


def wrap_environment(env, diversity_weight=200.0):
    """
    Convenience function to wrap environment
    
    Args:
        env: Your EnhancedHoneypotEnvironment instance
        diversity_weight: How strongly to enforce diversity (higher = more enforcement)
    
    Returns:
        Wrapped environment
    """
    return DiversityEnforcingWrapper(env, window_size=20, diversity_weight=diversity_weight)


# Test the wrapper
if __name__ == "__main__":
    print("Diversity Enforcing Wrapper")
    print("=" * 60)
    print("\nThis wrapper adds strong diversity enforcement at the")
    print("environment level, complementing reward shaping.")
    print("\nKey features:")
    print("  • Penalties for overused actions (>20% usage)")
    print("  • Harsh penalties for repetitive sequences")
    print("  • Bonuses for using underutilized actions")
    print("  • Entropy-based episode rewards")
    print("\nUsage in train_agent_enhanced.py:")
    print("  from diversity_enforcing_wrapper import wrap_environment")
    print("  env = EnhancedHoneypotEnvironment()")
    print("  env = wrap_environment(env, diversity_weight=200.0)")
    print("\nThen train normally!")#!/usr/bin/env python3
"""
Environment Wrapper with Diversity Enforcement
This wraps your existing environment to force action diversity
"""
import numpy as np
from collections import deque


class DiversityEnforcingWrapper:
    """
    Wraps honeypot environment to enforce action diversity
    Uses a combination of:
    1. Action history tracking
    2. Strong penalties for repetitive behavior
    3. Bonus rewards for diverse action sequences
    """
    
    def __init__(self, env, window_size=20, diversity_weight=300.0, hard_cap=0.25):
        """
        Args:
            env: Base honeypot environment
            window_size: Look back this many actions
            diversity_weight: Strength of diversity enforcement
            hard_cap: Maximum allowed frequency for any action (0.25 = 25%)
        """
        self.env = env
        self.window_size = window_size
        self.diversity_weight = diversity_weight
        self.hard_cap = hard_cap
        
        # Track recent actions
        self.action_history = deque(maxlen=window_size)
        self.global_action_counts = np.zeros(env.action_dim)
        self.episode_action_counts = np.zeros(env.action_dim)
        
        # Pass through attributes
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.attack_types = env.attack_types
        self.actions = env.actions
        self.current_attack_type = 0
        
    def reset(self):
        """Reset environment and diversity tracking"""
        self.episode_action_counts = np.zeros(self.action_dim)
        state = self.env.reset()
        self.current_attack_type = self.env.current_attack_type
        return state
    
    def step(self, action):
        """Step with diversity-aware reward modification"""
        # Take action in base environment
        next_state, base_reward, done, info = self.env.step(action)
        
        # Track actions
        self.action_history.append(action)
        self.global_action_counts[action] += 1
        self.episode_action_counts[action] += 1
        
        # Calculate diversity-modified reward
        reward = self._modify_reward(base_reward, action, done)
        
        return next_state, reward, done, info
    
    def _modify_reward(self, base_reward, action, done):
        """Apply strong diversity enforcement"""
        reward = float(base_reward)
        
        # 1. Penalize if this action is overused globally
        total_global = self.global_action_counts.sum()
        if total_global > 100:
            global_freq = self.global_action_counts[action] / total_global
            expected_freq = 1.0 / self.action_dim
            
            # Strong penalty for overused actions
            if global_freq > expected_freq * 1.2:  # Over 20%
                overuse_factor = global_freq / expected_freq
                penalty = self.diversity_weight * (overuse_factor - 1.2)
                reward -= penalty
                
                # Extra harsh if really overused
                if global_freq > expected_freq * 2.0:  # Over 33%
                    reward *= 0.5  # Cut reward in half
        
        # 2. Penalize repetitive recent actions
        if len(self.action_history) >= 5:
            recent_5 = list(self.action_history)[-5:]
            action_count = recent_5.count(action)
            
            # Penalty grows exponentially with repetition
            if action_count >= 3:
                reward -= self.diversity_weight * (action_count - 2)
            
            # Bonus for using new action
            if action_count == 1:
                reward += self.diversity_weight * 0.5
        
        # 3. Bonus for balanced episode-level distribution
        total_episode = self.episode_action_counts.sum()
        if done and total_episode > 0:
            episode_freqs = self.episode_action_counts / total_episode
            
            # Calculate entropy (measure of diversity)
            nonzero_freqs = episode_freqs[episode_freqs > 0]
            if len(nonzero_freqs) > 0:
                entropy = -np.sum(nonzero_freqs * np.log(nonzero_freqs + 1e-10))
                max_entropy = np.log(self.action_dim)
                
                # Big bonus for high entropy (diverse actions)
                diversity_ratio = entropy / max_entropy
                reward += self.diversity_weight * diversity_ratio * 2.0
        
        # 4. Reward for using underutilized actions
        if total_global > 100:
            global_freq = self.global_action_counts[action] / total_global
            expected_freq = 1.0 / self.action_dim
            
            if global_freq < expected_freq * 0.8:  # Under 13%
                underuse_bonus = self.diversity_weight * (expected_freq * 0.8 - global_freq)
                reward += underuse_bonus
        
        return reward
    
    def get_diversity_stats(self):
        """Get current diversity statistics"""
        total = self.global_action_counts.sum()
        if total == 0:
            return None
        
        freqs = self.global_action_counts / total
        entropy = -np.sum(freqs[freqs > 0] * np.log(freqs[freqs > 0] + 1e-10))
        max_entropy = np.log(self.action_dim)
        
        return {
            'action_counts': self.global_action_counts.tolist(),
            'action_frequencies': freqs.tolist(),
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'diversity_percent': float(entropy / max_entropy * 100)
        }


def wrap_environment(env, diversity_weight=200.0):
    """
    Convenience function to wrap environment
    
    Args:
        env: Your EnhancedHoneypotEnvironment instance
        diversity_weight: How strongly to enforce diversity (higher = more enforcement)
    
    Returns:
        Wrapped environment
    """
    return DiversityEnforcingWrapper(env, window_size=20, diversity_weight=diversity_weight)


# Test the wrapper
if __name__ == "__main__":
    print("Diversity Enforcing Wrapper")
    print("=" * 60)
    print("\nThis wrapper adds strong diversity enforcement at the")
    print("environment level, complementing reward shaping.")
    print("\nKey features:")
    print("  • Penalties for overused actions (>20% usage)")
    print("  • Harsh penalties for repetitive sequences")
    print("  • Bonuses for using underutilized actions")
    print("  • Entropy-based episode rewards")
    print("\nUsage in train_agent_enhanced.py:")
    print("  from diversity_enforcing_wrapper import wrap_environment")
    print("  env = EnhancedHoneypotEnvironment()")
    print("  env = wrap_environment(env, diversity_weight=200.0)")
    print("\nThen train normally!")
