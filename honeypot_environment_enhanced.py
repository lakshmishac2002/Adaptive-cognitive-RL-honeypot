import numpy as np
import random
from datetime import datetime
import json

class EnhancedHoneypotEnvironment:
    """
    Enhanced Honeypot Environment with 9 specific attack types
    """
    
    def __init__(self):
        # 9 Attack Types
        self.attack_types = {
            0: 'port_scan',
            1: 'ssh_bruteforce', 
            2: 'sql_injection',
            3: 'xss',
            4: 'ddos',
            5: 'malware',
            6: 'directory_traversal',
            7: 'command_injection',
            8: 'zero_day'
        }
        
        # 6 Response Actions
        self.actions = {
            0: 'respond_realistic',    # Act like real system
            1: 'delay_response',        # Slow down attacker
            2: 'fake_error',            # Show fake error
            3: 'fake_data',             # Provide honeypot data
            4: 'redirect_deep',         # Redirect to deeper honeypot
            5: 'block'                  # Block the IP
        }
        
        self.action_dim = len(self.actions)
        self.state_dim = 20  # Enhanced state representation
        
        # Attack characteristics
        self.attack_signatures = self._initialize_attack_signatures()
        
        # Episode state
        self.current_attack_type = None
        self.current_step = 0
        self.max_steps = 500
        self.attacker_engaged_time = 0
        self.attacker_suspicious_level = 0.0
        self.intelligence_gathered = 0
        
    def _initialize_attack_signatures(self):
        """Define characteristics for each attack type"""
        return {
            'port_scan': {
                'frequency': 'high',        # Many requests
                'payload_size': 'small',    # Small packets
                'timing': 'regular',        # Consistent timing
                'stealth': 0.7,             # Somewhat stealthy
                'duration': 'short',        # Quick scans
                'detection_difficulty': 0.3 # Easy to detect
            },
            'ssh_bruteforce': {
                'frequency': 'very_high',
                'payload_size': 'small',
                'timing': 'regular',
                'stealth': 0.3,
                'duration': 'long',
                'detection_difficulty': 0.2  # Very easy
            },
            'sql_injection': {
                'frequency': 'medium',
                'payload_size': 'medium',
                'timing': 'irregular',
                'stealth': 0.6,
                'duration': 'medium',
                'detection_difficulty': 0.4
            },
            'xss': {
                'frequency': 'low',
                'payload_size': 'medium',
                'timing': 'irregular',
                'stealth': 0.7,
                'duration': 'short',
                'detection_difficulty': 0.5
            },
            'ddos': {
                'frequency': 'extreme',
                'payload_size': 'large',
                'timing': 'rapid',
                'stealth': 0.1,
                'duration': 'long',
                'detection_difficulty': 0.1  # Obvious
            },
            'malware': {
                'frequency': 'low',
                'payload_size': 'large',
                'timing': 'single',
                'stealth': 0.8,
                'duration': 'instant',
                'detection_difficulty': 0.6
            },
            'directory_traversal': {
                'frequency': 'medium',
                'payload_size': 'small',
                'timing': 'irregular',
                'stealth': 0.6,
                'duration': 'medium',
                'detection_difficulty': 0.4
            },
            'command_injection': {
                'frequency': 'medium',
                'payload_size': 'medium',
                'timing': 'irregular',
                'stealth': 0.7,
                'duration': 'medium',
                'detection_difficulty': 0.5
            },
            'zero_day': {
                'frequency': 'low',
                'payload_size': 'variable',
                'timing': 'irregular',
                'stealth': 0.9,
                'duration': 'variable',
                'detection_difficulty': 0.9  # Very hard
            }
        }
    
    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.attacker_engaged_time = 0
        self.attacker_suspicious_level = 0.0
        self.intelligence_gathered = 0
        
        # Randomly select attack type
        self.current_attack_type = random.choice(list(self.attack_types.keys()))
        
        return self._get_state()
    
    def _get_state(self):
        """
        Generate state representation (20 features)
        
        Features:
        0: Attack type (one-hot encoded will be 0-8, normalized)
        1: Attack frequency (0-1)
        2: Payload size (0-1)
        3: Timing pattern (0-1)
        4: Stealth level (0-1)
        5: Detection difficulty (0-1)
        6: Current step / max_steps
        7: Attacker engaged time (normalized)
        8: Attacker suspicious level (0-1)
        9: Intelligence gathered (normalized)
        10-14: Attack pattern features
        15-19: Environmental features
        """
        state = np.zeros(20)
        
        attack_name = self.attack_types[self.current_attack_type]
        sig = self.attack_signatures[attack_name]
        
        # Basic features
        state[0] = self.current_attack_type / 8.0  # Normalize
        state[1] = self._encode_frequency(sig['frequency'])
        state[2] = self._encode_size(sig['payload_size'])
        state[3] = self._encode_timing(sig['timing'])
        state[4] = sig['stealth']
        state[5] = sig['detection_difficulty']
        state[6] = self.current_step / self.max_steps
        state[7] = min(self.attacker_engaged_time / 100.0, 1.0)
        state[8] = self.attacker_suspicious_level
        state[9] = min(self.intelligence_gathered / 50.0, 1.0)
        
        # Attack pattern features (random variations for realism)
        state[10] = random.uniform(0.3, 0.9)  # Request rate
        state[11] = random.uniform(0.2, 0.8)  # Payload entropy
        state[12] = random.uniform(0.1, 0.7)  # Header anomalies
        state[13] = random.uniform(0.4, 0.9)  # Protocol compliance
        state[14] = random.uniform(0.2, 0.6)  # Source reputation
        
        # Environmental features
        state[15] = (datetime.now().hour / 24.0)  # Time of day
        state[16] = random.uniform(0.3, 0.8)      # System load
        state[17] = random.uniform(0.5, 1.0)      # Network latency
        state[18] = random.uniform(0.0, 0.5)      # Previous attack similarity
        state[19] = random.uniform(0.2, 0.7)      # Honeypot believability
        
        return state
    
    def _encode_frequency(self, freq):
        mapping = {'low': 0.2, 'medium': 0.5, 'high': 0.7, 'very_high': 0.9, 'extreme': 1.0}
        return mapping.get(freq, 0.5)
    
    def _encode_size(self, size):
        mapping = {'small': 0.2, 'medium': 0.5, 'large': 0.8, 'variable': 0.5}
        return mapping.get(size, 0.5)
    
    def _encode_timing(self, timing):
        mapping = {'single': 0.1, 'irregular': 0.5, 'regular': 0.7, 'rapid': 0.9}
        return mapping.get(timing, 0.5)
    
    def step(self, action):
        """Execute action and return next state, reward, done, info"""
        self.current_step += 1
        
        # Calculate reward based on action effectiveness for this attack type
        reward = self._calculate_reward(action)
        
        # Update attacker state based on action
        self._update_attacker_state(action)
        
        # Get next state
        next_state = self._get_state()
        
        # Episode done conditions
        done = (self.current_step >= self.max_steps or 
                self.attacker_suspicious_level >= 1.0)
        
        info = {
            'attack_type': self.attack_types[self.current_attack_type],
            'action_taken': self.actions[action],
            'step': self.current_step,
            'engaged_time': self.attacker_engaged_time,
            'suspicious_level': self.attacker_suspicious_level,
            'intelligence': self.intelligence_gathered
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action):
        """Calculate reward based on action effectiveness for current attack"""
        reward = 0
        
        attack_name = self.attack_types[self.current_attack_type]
        action_name = self.actions[action]
        
        # Reward matrix: Best actions for each attack type
        optimal_actions = {
            'port_scan': ['respond_realistic', 'delay_response'],
            'ssh_bruteforce': ['fake_data', 'delay_response'],
            'sql_injection': ['fake_error', 'fake_data'],
            'xss': ['respond_realistic', 'fake_error'],
            'ddos': ['block', 'delay_response'],
            'malware': ['fake_data', 'redirect_deep'],
            'directory_traversal': ['fake_error', 'redirect_deep'],
            'command_injection': ['fake_data', 'fake_error'],
            'zero_day': ['respond_realistic', 'redirect_deep']
        }
        
        # Base reward for optimal action
        if action_name in optimal_actions[attack_name]:
            reward += 15  # High reward for optimal response
            self.intelligence_gathered += 3
            self.attacker_engaged_time += 2
        else:
            reward += 5  # Still some reward for any action
            self.intelligence_gathered += 1
            self.attacker_engaged_time += 1
        
        # Bonus for keeping attacker engaged
        reward += self.attacker_engaged_time * 0.3
        
        # Bonus for intelligence gathering
        reward += self.intelligence_gathered * 0.5
        
        # Penalty for making honeypot obvious
        if self.attacker_suspicious_level > 0.7:
            reward -= 10
        
        # Penalty for early blocking (lost intelligence opportunity)
        if action_name == 'block' and self.attacker_engaged_time < 10:
            reward -= 8
        
        # Bonus for appropriate blocking at right time
        if action_name == 'block' and self.attacker_engaged_time > 20:
            reward += 5
        
        # Attack-specific rewards
        if attack_name == 'ddos' and action_name == 'block':
            reward += 10  # DDoS should be blocked quickly
        
        if attack_name == 'ssh_bruteforce' and action_name == 'fake_data':
            reward += 8  # Provide fake credentials
        
        if attack_name == 'sql_injection' and action_name == 'fake_error':
            reward += 8  # Fake DB errors are effective
        
        if attack_name == 'zero_day' and action_name == 'respond_realistic':
            reward += 12  # Critical to not reveal it's a honeypot
        
        return reward
    
    def _update_attacker_state(self, action):
        """Update attacker's state based on action taken"""
        action_name = self.actions[action]
        attack_name = self.attack_types[self.current_attack_type]
        
        # Different actions affect suspicion differently
        suspicion_increase = {
            'respond_realistic': 0.01,
            'delay_response': 0.02,
            'fake_error': 0.03,
            'fake_data': 0.02,
            'redirect_deep': 0.04,
            'block': 0.10  # Blocking makes it obvious
        }
        
        self.attacker_suspicious_level += suspicion_increase.get(action_name, 0.02)
        
        # Some attacks are more likely to detect honeypots
        if attack_name in ['zero_day', 'malware']:
            self.attacker_suspicious_level += 0.01
        
        # Cap suspicious level
        self.attacker_suspicious_level = min(self.attacker_suspicious_level, 1.0)
    
    def get_attack_statistics(self):
        """Return statistics about current attack"""
        attack_name = self.attack_types[self.current_attack_type]
        return {
            'attack_type': attack_name,
            'engaged_time': self.attacker_engaged_time,
            'intelligence_gathered': self.intelligence_gathered,
            'suspicious_level': self.attacker_suspicious_level,
            'detected': self.attacker_suspicious_level > 0.8,
            'signature': self.attack_signatures[attack_name]
        }


# Test the environment
if __name__ == "__main__":
    print("=" * 60)
    print("Enhanced Honeypot Environment - Attack Coverage Test")
    print("=" * 60)
    print()
    
    env = EnhancedHoneypotEnvironment()
    
    print(" Supported Attack Types:")
    for idx, attack in env.attack_types.items():
        sig = env.attack_signatures[attack]
        print(f"  {idx+1}. {attack.upper()}")
        print(f"     - Detection Difficulty: {sig['detection_difficulty']*100:.0f}%")
        print(f"     - Stealth Level: {sig['stealth']*100:.0f}%")
    
    print()
    print(" Available Response Actions:")
    for idx, action in env.actions.items():
        print(f"  {idx}. {action}")
    
    print()
    print(" Running test episode...")
    state = env.reset()
    print(f"   Attack Type: {env.attack_types[env.current_attack_type]}")
    print(f"   State Dimension: {len(state)}")
    
    total_reward = 0
    for step in range(10):
        action = random.randint(0, env.action_dim - 1)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if step < 3:  # Show first 3 steps
            print(f"   Step {step+1}: Action={info['action_taken']}, Reward={reward:.2f}")
    
    stats = env.get_attack_statistics()
    print()
    print(" Episode Statistics:")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Engaged Time: {stats['engaged_time']}")
    print(f"   Intelligence: {stats['intelligence_gathered']}")
    print(f"   Detected: {'Yes' if stats['detected'] else 'No'}")
    
    print()
    print(" All 9 attack types are covered!")
    print("=" * 60)
