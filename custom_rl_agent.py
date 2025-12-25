#!/usr/bin/env python3
"""
Ultimate Fixed Agent
Maintains diversity throughout training with adaptive mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import random


class ImprovedDuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):  # Smaller network!
        super(ImprovedDuelingDQN, self).__init__()
        
        self.action_dim = action_dim
        
        # Smaller feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage streams
        self.advantage_streams = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(action_dim)
        ])
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.3)  # Even smaller!
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        features = self.feature_layer(state)
        value = self.value_stream(features)
        
        advantages = torch.stack([
            stream(features) for stream in self.advantage_streams
        ], dim=1).squeeze(-1)
        
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        # Aggressive normalization
        q_values = torch.tanh(q_values / 25.0) * 25.0  # Tighter bounds!
        
        return q_values


class AdaptivePrioritizedReplayBuffer:
    def __init__(self, capacity=30000):  # Smaller buffer
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = 0.6
        self.beta_start = 0.4
        self.beta_frames = 50000
        self.frame = 1
    
    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class FixedHoneypotRLAgent:
    """Agent with strong diversity maintenance"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.0002,  # Lower LR
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.20,  # Higher minimum!
        epsilon_decay=0.995,
        target_update_freq=300,  # More frequent
        entropy_coef=0.02,  # Higher entropy
        diversity_bonus_weight=15.0
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.entropy_coef = entropy_coef
        self.diversity_bonus_weight = diversity_bonus_weight
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Smaller networks
        self.policy_net = ImprovedDuelingDQN(state_dim, action_dim, hidden_dim=128).to(self.device)
        self.target_net = ImprovedDuelingDQN(state_dim, action_dim, hidden_dim=128).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-3  # Strong regularization
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=2000, gamma=0.95
        )
        
        self.memory = AdaptivePrioritizedReplayBuffer(capacity=30000)
        
        # Tracking
        self.training_step = 0
        self.action_counts = np.zeros(action_dim)
        self.action_rewards = defaultdict(list)
        self.episode_actions = []
        
        # Adaptive diversity mechanism
        self.recent_action_usage = deque(maxlen=1000)  # Last 1000 actions
        self.eval_action_counts = np.zeros(action_dim)
        self.eval_epsilon = 0.15  # Higher eval epsilon
        
    def get_diversity_bonus(self, training=True):
        """Calculate adaptive diversity bonus"""
        counts = self.action_counts if training else self.eval_action_counts
        
        if counts.sum() < 20:
            return np.zeros(self.action_dim)
        
        frequencies = counts / counts.sum()
        expected = 1.0 / self.action_dim
        
        # Stronger bonus for underused, stronger penalty for overused
        bonus = np.zeros(self.action_dim)
        for i in range(self.action_dim):
            if frequencies[i] < expected * 0.8:  # Underused
                bonus[i] = self.diversity_bonus_weight * (expected * 0.8 - frequencies[i]) * 100
            elif frequencies[i] > expected * 1.3:  # Overused
                bonus[i] = -self.diversity_bonus_weight * (frequencies[i] - expected * 1.3) * 50
        
        return bonus
    
    def select_action(self, state, training=True):
        """Action selection with strong diversity pressure"""
        epsilon = self.epsilon if training else self.eval_epsilon
        
        # Higher exploration rate if diversity is low
        counts = self.action_counts if training else self.eval_action_counts
        if counts.sum() > 100:
            max_freq = counts.max() / counts.sum()
            if max_freq > 0.3:  # If any action > 30%, increase exploration
                epsilon = min(epsilon * 1.5, 0.4)
        
        if random.random() < epsilon:
            # Inverse frequency sampling
            if counts.sum() > 20:
                probs = 1.0 / (counts + 0.5)
                probs = probs / probs.sum()
                action = np.random.choice(self.action_dim, p=probs)
            else:
                action = random.randrange(self.action_dim)
        else:
            # Greedy with diversity bonus
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                
                # Add diversity bonus
                diversity_bonus = self.get_diversity_bonus(training)
                q_values = q_values + diversity_bonus
                
                action = np.argmax(q_values)
        
        # Track
        if training:
            self.action_counts[action] += 1
            self.recent_action_usage.append(action)
        else:
            self.eval_action_counts[action] += 1
        
        return int(action)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.action_rewards[action].append(reward)
        self.episode_actions.append(action)
    
    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # TD error
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        
        # Huber loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        loss = (loss * weights).mean()
        
        # Stronger entropy regularization
        q_values_all = self.policy_net(states)
        probs = F.softmax(q_values_all / 1.0, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(1).mean()
        total_loss = loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)  # Stricter!
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Update target
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return total_loss.item()
    
    def reset_eval_counts(self):
        self.eval_action_counts = np.zeros(self.action_dim)
    
    def get_action_statistics(self):
        total = self.action_counts.sum()
        if total == 0:
            return {}
        
        percentages = (self.action_counts / total) * 100
        
        avg_rewards = {}
        for action in range(self.action_dim):
            rewards = self.action_rewards.get(action, [])
            avg_rewards[action] = np.mean(rewards) if rewards else 0.0
        
        probs = self.action_counts / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(self.action_dim)
        
        return {
            'action_counts': self.action_counts.tolist(),
            'action_percentages': percentages.tolist(),
            'action_avg_rewards': avg_rewards,
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'diversity_percent': float(entropy / max_entropy * 100)
        }
    
    def reset_episode_tracking(self):
        self.episode_actions = []
    
    def save(self, filepath):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'action_counts': self.action_counts,
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']
        self.action_counts = checkpoint['action_counts']


if __name__ == "__main__":
    print("=" * 70)
    print("ULTIMATE FIXED AGENT")
    print("=" * 70)
    print("\nKey improvements:")
    print("  • Adaptive diversity bonus (stronger penalties)")
    print("  • Higher minimum epsilon (0.20 vs 0.15)")
    print("  • Adaptive exploration (increases if diversity drops)")
    print("  • Smaller network (128 vs 256 hidden dims)")
    print("  • Stronger regularization")
    print("  • Higher eval epsilon (0.15)")
    print("  • Stricter gradient clipping (0.5 vs 1.0)")
    print("=" * 70)
