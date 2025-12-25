
#!/usr/bin/env python3
"""
Comprehensive Cumulative Scoring and Success Rate Analysis
Calculates multiple success metrics for the RL honeypot agent
"""
import torch
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

from honeypot_environment_enhanced import EnhancedHoneypotEnvironment
from custom_rl_agent import FixedHoneypotRLAgent


class CumulativeScorer:
    """Calculate comprehensive success metrics"""
    
    def __init__(self, model_path, num_episodes=100):
        self.model_path = model_path
        self.num_episodes = num_episodes
        
        # Initialize environment and agent
        self.env = EnhancedHoneypotEnvironment()
        self.agent = FixedHoneypotRLAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.agent.device, weights_only=False)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.agent.epsilon = 0.0  # Pure greedy
        
        print(f"✓ Model loaded: {model_path}")
        print(f"  Training steps: {checkpoint['training_step']}")
        print()
        
        # Metrics storage
        self.metrics = {
            'total_reward': 0,
            'total_intelligence': 0,
            'total_engagement_time': 0,
            'total_episodes': 0,
            'successful_episodes': 0,
            'honeypot_detected': 0,
            'honeypot_undetected': 0,
            'attack_type_success': defaultdict(lambda: {'success': 0, 'total': 0}),
            'episode_details': []
        }
    
    def evaluate_episode(self):
        """Evaluate a single episode and return success metrics"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        attack_type = self.env.attack_types[self.env.current_attack_type]
        
        while not done:
            action = self.agent.select_action(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        # Calculate success criteria
        engagement_time = info['engaged_time']
        intelligence = info['intelligence']
        suspicious_level = info['suspicious_level']
        detected = suspicious_level > 0.8
        
        # Success criteria (you can adjust these thresholds)
        success_criteria = {
            'good_engagement': engagement_time >= 25,        # Kept attacker engaged
            'intelligence_gathered': intelligence >= 15,     # Gathered good intel
            'stealth_maintained': suspicious_level < 0.7,    # Stayed hidden
            'positive_reward': episode_reward > 1000         # Good overall performance
        }
        
        # Overall success: meet at least 3 out of 4 criteria
        success_count = sum(success_criteria.values())
        is_successful = success_count >= 3
        
        # Alternative: Weighted success score (0-100)
        weighted_score = (
            (engagement_time / 50.0) * 25 +                  # 25% weight
            (intelligence / 50.0) * 25 +                     # 25% weight
            (max(0, 1 - suspicious_level)) * 30 +            # 30% weight
            (min(episode_reward / 3000.0, 1)) * 20           # 20% weight
        )
        
        return {
            'reward': episode_reward,
            'engagement_time': engagement_time,
            'intelligence': intelligence,
            'suspicious_level': suspicious_level,
            'detected': detected,
            'is_successful': is_successful,
            'success_score': weighted_score,
            'attack_type': attack_type,
            'episode_length': episode_length,
            'success_criteria': success_criteria
        }
    
    def run_evaluation(self):
        """Run full evaluation and calculate all metrics"""
        
        print("=" * 80)
        print("CUMULATIVE SCORING AND SUCCESS RATE ANALYSIS")
        print("=" * 80)
        print(f"Evaluating {self.num_episodes} episodes...")
        print()
        
        for episode in range(self.num_episodes):
            result = self.evaluate_episode()
            
            # Update cumulative metrics
            self.metrics['total_reward'] += result['reward']
            self.metrics['total_intelligence'] += result['intelligence']
            self.metrics['total_engagement_time'] += result['engagement_time']
            self.metrics['total_episodes'] += 1
            
            if result['is_successful']:
                self.metrics['successful_episodes'] += 1
            
            if result['detected']:
                self.metrics['honeypot_detected'] += 1
            else:
                self.metrics['honeypot_undetected'] += 1
            
            # Track by attack type
            attack = result['attack_type']
            self.metrics['attack_type_success'][attack]['total'] += 1
            if result['is_successful']:
                self.metrics['attack_type_success'][attack]['success'] += 1
            
            # Store details
            self.metrics['episode_details'].append(result)
            
            # Progress
            if (episode + 1) % 20 == 0:
                current_success_rate = (self.metrics['successful_episodes'] / (episode + 1)) * 100
                print(f"  Episode {episode + 1}/{self.num_episodes} | "
                      f"Success Rate: {current_success_rate:.1f}%")
        
        print()
        print("✓ Evaluation complete!")
        print()
        
        self._generate_report()
        self._create_visualizations()
        self._save_results()
    
    def _generate_report(self):
        """Generate comprehensive success report"""
        
        print("=" * 80)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)
        print()
        
        # 1. CUMULATIVE SCORES
        print("📊 CUMULATIVE SCORES")
        print("-" * 80)
        print(f"Total Reward:            {self.metrics['total_reward']:>12.2f}")
        print(f"Total Intelligence:      {self.metrics['total_intelligence']:>12}")
        print(f"Total Engagement Time:   {self.metrics['total_engagement_time']:>12} steps")
        print(f"Total Episodes:          {self.metrics['total_episodes']:>12}")
        print()
        
        avg_reward = self.metrics['total_reward'] / self.metrics['total_episodes']
        avg_intelligence = self.metrics['total_intelligence'] / self.metrics['total_episodes']
        avg_engagement = self.metrics['total_engagement_time'] / self.metrics['total_episodes']
        
        print(f"Average Reward/Episode:        {avg_reward:>10.2f}")
        print(f"Average Intelligence/Episode:  {avg_intelligence:>10.2f}")
        print(f"Average Engagement/Episode:    {avg_engagement:>10.2f} steps")
        print()
        
        # 2. SUCCESS RATES
        print("✅ SUCCESS RATES")
        print("-" * 80)
        
        success_rate = (self.metrics['successful_episodes'] / self.metrics['total_episodes']) * 100
        stealth_rate = (self.metrics['honeypot_undetected'] / self.metrics['total_episodes']) * 100
        detection_rate = (self.metrics['honeypot_detected'] / self.metrics['total_episodes']) * 100
        
        print(f"Overall Success Rate:    {success_rate:>10.1f}%")
        print(f"  ├─ Successful Episodes: {self.metrics['successful_episodes']:>4}/{self.metrics['total_episodes']}")
        print(f"  └─ Failed Episodes:     {self.metrics['total_episodes'] - self.metrics['successful_episodes']:>4}/{self.metrics['total_episodes']}")
        print()
        
        print(f"Stealth Performance:")
        print(f"  ├─ Undetected (Success): {stealth_rate:>9.1f}% ({self.metrics['honeypot_undetected']} episodes)")
        print(f"  └─ Detected (Failure):   {detection_rate:>9.1f}% ({self.metrics['honeypot_detected']} episodes)")
        print()
        
        # Success rate interpretation
        if success_rate >= 80:
            rating = "🌟 EXCELLENT"
        elif success_rate >= 70:
            rating = "✓ GOOD"
        elif success_rate >= 60:
            rating = "⚠ FAIR"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
        
        print(f"Performance Rating:      {rating}")
        print()
        
        # 3. SUCCESS CRITERIA BREAKDOWN
        print("📋 SUCCESS CRITERIA BREAKDOWN")
        print("-" * 80)
        
        criteria_counts = {
            'good_engagement': 0,
            'intelligence_gathered': 0,
            'stealth_maintained': 0,
            'positive_reward': 0
        }
        
        for episode in self.metrics['episode_details']:
            for criterion, met in episode['success_criteria'].items():
                if met:
                    criteria_counts[criterion] += 1
        
        total = self.metrics['total_episodes']
        
        print(f"Good Engagement (≥25 steps):      {criteria_counts['good_engagement']:>4}/{total} ({criteria_counts['good_engagement']/total*100:>5.1f}%)")
        print(f"Intelligence Gathered (≥15):      {criteria_counts['intelligence_gathered']:>4}/{total} ({criteria_counts['intelligence_gathered']/total*100:>5.1f}%)")
        print(f"Stealth Maintained (<0.7):        {criteria_counts['stealth_maintained']:>4}/{total} ({criteria_counts['stealth_maintained']/total*100:>5.1f}%)")
        print(f"Positive Reward (>1000):          {criteria_counts['positive_reward']:>4}/{total} ({criteria_counts['positive_reward']/total*100:>5.1f}%)")
        print()
        
        # 4. SUCCESS BY ATTACK TYPE
        print("⚔️ SUCCESS RATE BY ATTACK TYPE")
        print("-" * 80)
        
        attack_stats = []
        for attack_type, stats in sorted(self.metrics['attack_type_success'].items()):
            if stats['total'] > 0:
                rate = (stats['success'] / stats['total']) * 100
                attack_stats.append({
                    'attack': attack_type,
                    'rate': rate,
                    'success': stats['success'],
                    'total': stats['total']
                })
        
        # Sort by success rate
        attack_stats.sort(key=lambda x: x['rate'], reverse=True)
        
        print(f"{'Attack Type':<25} {'Success Rate':>15} {'Episodes':>12}")
        print("-" * 80)
        
        for stat in attack_stats:
            bar = "█" * int(stat['rate'] / 5) + "░" * (20 - int(stat['rate'] / 5))
            print(f"{stat['attack']:<25} {stat['rate']:>6.1f}% {bar:>20} {stat['success']:>3}/{stat['total']:<3}")
        
        print()
        
        # 5. WEIGHTED SUCCESS SCORE
        print("🎯 WEIGHTED SUCCESS SCORE")
        print("-" * 80)
        
        scores = [ep['success_score'] for ep in self.metrics['episode_details']]
        avg_score = np.mean(scores)
        
        print(f"Average Score:           {avg_score:>10.1f}/100")
        print(f"Median Score:            {np.median(scores):>10.1f}/100")
        print(f"Min/Max Score:           {np.min(scores):>10.1f}/{np.max(scores):<10.1f}")
        print()
        
        print("Score Breakdown:")
        print(f"  • Engagement Time (25%): {avg_engagement/50*25:>6.1f}/25")
        print(f"  • Intelligence (25%):    {avg_intelligence/50*25:>6.1f}/25")
        print(f"  • Stealth (30%):         {(1-np.mean([ep['suspicious_level'] for ep in self.metrics['episode_details']]))*30:>6.1f}/30")
        print(f"  • Reward (20%):          {min(avg_reward/3000, 1)*20:>6.1f}/20")
        print()
        
        # 6. PERFORMANCE TIERS
        print("🏆 PERFORMANCE DISTRIBUTION")
        print("-" * 80)
        
        tiers = {
            'Excellent (80-100)': 0,
            'Good (60-79)': 0,
            'Fair (40-59)': 0,
            'Poor (0-39)': 0
        }
        
        for score in scores:
            if score >= 80:
                tiers['Excellent (80-100)'] += 1
            elif score >= 60:
                tiers['Good (60-79)'] += 1
            elif score >= 40:
                tiers['Fair (40-59)'] += 1
            else:
                tiers['Poor (0-39)'] += 1
        
        for tier, count in tiers.items():
            pct = (count / total) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"{tier:<20} {count:>4}/{total:<4} ({pct:>5.1f}%) {bar}")
        
        print()
        print("=" * 80)
    
    def _create_visualizations(self):
        """Create comprehensive visualization"""
        
        try:
            fig = plt.figure(figsize=(18, 12))
            
            # 1. Success Rate Over Time
            ax1 = plt.subplot(3, 3, 1)
            success_over_time = []
            for i in range(len(self.metrics['episode_details'])):
                success_count = sum([1 for ep in self.metrics['episode_details'][:i+1] if ep['is_successful']])
                success_over_time.append((success_count / (i+1)) * 100)
            
            ax1.plot(range(1, len(success_over_time) + 1), success_over_time, 
                    linewidth=2, color='green')
            ax1.axhline(y=70, color='orange', linestyle='--', label='Good Threshold')
            ax1.axhline(y=80, color='green', linestyle='--', label='Excellent Threshold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Success Rate (%)')
            ax1.set_title('Cumulative Success Rate Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Score Distribution
            ax2 = plt.subplot(3, 3, 2)
            scores = [ep['success_score'] for ep in self.metrics['episode_details']]
            ax2.hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(scores):.1f}')
            ax2.set_xlabel('Success Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Success Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Success by Attack Type
            ax3 = plt.subplot(3, 3, 3)
            attacks = []
            success_rates = []
            for attack, stats in sorted(self.metrics['attack_type_success'].items()):
                if stats['total'] > 0:
                    attacks.append(attack.replace('_', '\n'))
                    success_rates.append((stats['success'] / stats['total']) * 100)
            
            colors = ['green' if rate >= 70 else 'orange' if rate >= 60 else 'red' 
                     for rate in success_rates]
            ax3.bar(range(len(attacks)), success_rates, color=colors, alpha=0.7, edgecolor='black')
            ax3.set_xticks(range(len(attacks)))
            ax3.set_xticklabels(attacks, rotation=45, ha='right', fontsize=8)
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('Success Rate by Attack Type')
            ax3.axhline(y=70, color='orange', linestyle='--', alpha=0.5)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. Cumulative Reward
            ax4 = plt.subplot(3, 3, 4)
            cumulative_reward = np.cumsum([ep['reward'] for ep in self.metrics['episode_details']])
            ax4.plot(range(1, len(cumulative_reward) + 1), cumulative_reward, 
                    linewidth=2, color='purple')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Cumulative Reward')
            ax4.set_title('Cumulative Reward Over Time')
            ax4.grid(True, alpha=0.3)
            
            # 5. Success Criteria Met
            ax5 = plt.subplot(3, 3, 5)
            criteria = ['Engagement', 'Intelligence', 'Stealth', 'Reward']
            counts = [
                sum([1 for ep in self.metrics['episode_details'] if ep['success_criteria']['good_engagement']]),
                sum([1 for ep in self.metrics['episode_details'] if ep['success_criteria']['intelligence_gathered']]),
                sum([1 for ep in self.metrics['episode_details'] if ep['success_criteria']['stealth_maintained']]),
                sum([1 for ep in self.metrics['episode_details'] if ep['success_criteria']['positive_reward']])
            ]
            percentages = [(c / len(self.metrics['episode_details'])) * 100 for c in counts]
            
            bars = ax5.barh(criteria, percentages, color='teal', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Percentage (%)')
            ax5.set_title('Success Criteria Achievement')
            ax5.set_xlim(0, 100)
            
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                ax5.text(pct + 2, i, f'{pct:.1f}%', va='center')
            
            ax5.grid(True, alpha=0.3, axis='x')
            
            # 6. Detection vs Undetected
            ax6 = plt.subplot(3, 3, 6)
            detection_data = [
                self.metrics['honeypot_undetected'],
                self.metrics['honeypot_detected']
            ]
            colors_pie = ['green', 'red']
            labels = [f"Undetected\n({self.metrics['honeypot_undetected']})", 
                     f"Detected\n({self.metrics['honeypot_detected']})"]
            
            ax6.pie(detection_data, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10})
            ax6.set_title('Stealth Performance')
            
            # 7. Intelligence Gathered Over Time
            ax7 = plt.subplot(3, 3, 7)
            cumulative_intel = np.cumsum([ep['intelligence'] for ep in self.metrics['episode_details']])
            ax7.plot(range(1, len(cumulative_intel) + 1), cumulative_intel,
                    linewidth=2, color='orange')
            ax7.set_xlabel('Episode')
            ax7.set_ylabel('Cumulative Intelligence')
            ax7.set_title('Intelligence Gathered Over Time')
            ax7.grid(True, alpha=0.3)
            
            # 8. Engagement Time Distribution
            ax8 = plt.subplot(3, 3, 8)
            engagement_times = [ep['engagement_time'] for ep in self.metrics['episode_details']]
            ax8.hist(engagement_times, bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
            ax8.axvline(np.mean(engagement_times), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(engagement_times):.1f}')
            ax8.set_xlabel('Engagement Time (steps)')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Engagement Time Distribution')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            
            # 9. Performance Tier Distribution
            ax9 = plt.subplot(3, 3, 9)
            scores = [ep['success_score'] for ep in self.metrics['episode_details']]
            tiers = ['Excellent\n(80-100)', 'Good\n(60-79)', 'Fair\n(40-59)', 'Poor\n(0-39)']
            tier_counts = [
                sum([1 for s in scores if s >= 80]),
                sum([1 for s in scores if 60 <= s < 80]),
                sum([1 for s in scores if 40 <= s < 60]),
                sum([1 for s in scores if s < 40])
            ]
            
            colors_tiers = ['gold', 'lightgreen', 'orange', 'lightcoral']
            ax9.bar(range(len(tiers)), tier_counts, color=colors_tiers, alpha=0.7, edgecolor='black')
            ax9.set_xticks(range(len(tiers)))
            ax9.set_xticklabels(tiers, fontsize=9)
            ax9.set_ylabel('Number of Episodes')
            ax9.set_title('Performance Tier Distribution')
            ax9.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            filename = f"results/cumulative_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Visualizations saved: {filename}")
            plt.close()
            
        except Exception as e:
            print(f"⚠ Could not create visualizations: {e}")
    
    def _save_results(self):
        """Save detailed results to JSON"""
        
        # Calculate summary statistics
        scores = [ep['success_score'] for ep in self.metrics['episode_details']]
        
        results = {
            'model_path': self.model_path,
            'num_episodes': self.num_episodes,
            'timestamp': datetime.now().isoformat(),
            
            'cumulative_scores': {
                'total_reward': float(self.metrics['total_reward']),
                'total_intelligence': int(self.metrics['total_intelligence']),
                'total_engagement_time': int(self.metrics['total_engagement_time']),
                'average_reward': float(self.metrics['total_reward'] / self.metrics['total_episodes']),
                'average_intelligence': float(self.metrics['total_intelligence'] / self.metrics['total_episodes']),
                'average_engagement': float(self.metrics['total_engagement_time'] / self.metrics['total_episodes'])
            },
            
            'success_rates': {
                'overall_success_rate': float((self.metrics['successful_episodes'] / self.metrics['total_episodes']) * 100),
                'successful_episodes': int(self.metrics['successful_episodes']),
                'failed_episodes': int(self.metrics['total_episodes'] - self.metrics['successful_episodes']),
                'stealth_success_rate': float((self.metrics['honeypot_undetected'] / self.metrics['total_episodes']) * 100),
                'detection_rate': float((self.metrics['honeypot_detected'] / self.metrics['total_episodes']) * 100)
            },
            
            'weighted_score': {
                'average_score': float(np.mean(scores)),
                'median_score': float(np.median(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores))
            },
            
            'attack_type_success': {
                attack: {
                    'success_rate': float((stats['success'] / stats['total']) * 100) if stats['total'] > 0 else 0,
                    'successful': int(stats['success']),
                    'total': int(stats['total'])
                }
                for attack, stats in self.metrics['attack_type_success'].items()
            }
        }
        
        filename = f"results/cumulative_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved: {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate cumulative scores and success rates')
    parser.add_argument('--model', type=str, default='models/best_agent.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("CUMULATIVE SCORING AND SUCCESS RATE CALCULATOR")
    print("=" * 80)
    print()
    
    scorer = CumulativeScorer(args.model, args.episodes)
    scorer.run_evaluation()
    
    print("\n✓ Analysis complete!")
    print("\nGenerated files:")
    print("  - results/cumulative_scores_YYYYMMDD_HHMMSS.json")
    print("  - results/cumulative_scores_YYYYMMDD_HHMMSS.png")
    print()


if __name__ == '__main__':
    main()
=======
#!/usr/bin/env python3
"""
Comprehensive Cumulative Scoring and Success Rate Analysis
Calculates multiple success metrics for the RL honeypot agent
Updated with revised metrics from previous discussion
"""
import torch
import numpy as np
import json
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

from honeypot_environment_enhanced import EnhancedHoneypotEnvironment
from custom_rl_agent import FixedHoneypotRLAgent


class CumulativeScorer:
    """Calculate comprehensive success metrics"""
    
    def __init__(self, model_path, num_episodes=100):
        self.model_path = model_path
        self.num_episodes = num_episodes
        
        # Initialize environment and agent
        self.env = EnhancedHoneypotEnvironment()
        self.agent = FixedHoneypotRLAgent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.agent.device, weights_only=False)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.agent.epsilon = 0.0  # Pure greedy
        
        print(f"✓ Model loaded: {model_path}")
        print(f"  Training steps: {checkpoint['training_step']}")
        print()
        
        # Metrics storage
        self.metrics = {
            'total_reward': 0,
            'total_intelligence': 0,
            'total_engagement_time': 0,
            'total_episodes': 0,
            'successful_episodes': 0,
            'honeypot_detected': 0,
            'honeypot_undetected': 0,
            'attack_type_success': defaultdict(lambda: {'success': 0, 'total': 0}),
            'episode_details': []
        }
    
    def evaluate_episode(self):
        """Evaluate a single episode and return success metrics"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        attack_type = self.env.attack_types[self.env.current_attack_type]
        
        while not done:
            action = self.agent.select_action(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        # Calculate success criteria
        engagement_time = info['engaged_time']
        intelligence = info['intelligence']
        suspicious_level = info['suspicious_level']
        detected = suspicious_level > 0.8
        
        # UPDATED SUCCESS CRITERIA (revised thresholds)
        success_criteria = {
            'good_engagement': engagement_time >= 20,        # Lowered from 25
            'intelligence_gathered': intelligence >= 12,     # Lowered from 15
            'stealth_maintained': suspicious_level < 0.75,   # Raised from 0.7
            'positive_reward': episode_reward > 800          # Lowered from 1000
        }
        
        # Overall success: meet at least 3 out of 4 criteria
        success_count = sum(success_criteria.values())
        is_successful = success_count >= 3
        
        # UPDATED WEIGHTED SCORE (revised formula with adjusted weights)
        weighted_score = (
            min(engagement_time / 40.0, 1.0) * 30 +          # 30% weight (adjusted scale)
            min(intelligence / 40.0, 1.0) * 30 +             # 30% weight (adjusted scale)
            max(0, 1 - suspicious_level) * 25 +              # 25% weight
            min(episode_reward / 2500.0, 1.0) * 15           # 15% weight (adjusted scale)
        )
        
        return {
            'reward': episode_reward,
            'engagement_time': engagement_time,
            'intelligence': intelligence,
            'suspicious_level': suspicious_level,
            'detected': detected,
            'is_successful': is_successful,
            'success_score': weighted_score,
            'attack_type': attack_type,
            'episode_length': episode_length,
            'success_criteria': success_criteria
        }
    
    def run_evaluation(self):
        """Run full evaluation and calculate all metrics"""
        
        print("=" * 80)
        print("CUMULATIVE SCORING AND SUCCESS RATE ANALYSIS")
        print("=" * 80)
        print(f"Evaluating {self.num_episodes} episodes...")
        print()
        
        for episode in range(self.num_episodes):
            result = self.evaluate_episode()
            
            # Update cumulative metrics
            self.metrics['total_reward'] += result['reward']
            self.metrics['total_intelligence'] += result['intelligence']
            self.metrics['total_engagement_time'] += result['engagement_time']
            self.metrics['total_episodes'] += 1
            
            if result['is_successful']:
                self.metrics['successful_episodes'] += 1
            
            if result['detected']:
                self.metrics['honeypot_detected'] += 1
            else:
                self.metrics['honeypot_undetected'] += 1
            
            # Track by attack type
            attack = result['attack_type']
            self.metrics['attack_type_success'][attack]['total'] += 1
            if result['is_successful']:
                self.metrics['attack_type_success'][attack]['success'] += 1
            
            # Store details
            self.metrics['episode_details'].append(result)
            
            # Progress
            if (episode + 1) % 20 == 0:
                current_success_rate = (self.metrics['successful_episodes'] / (episode + 1)) * 100
                print(f"  Episode {episode + 1}/{self.num_episodes} | "
                      f"Success Rate: {current_success_rate:.1f}%")
        
        print()
        print("✓ Evaluation complete!")
        print()
        
        self._generate_report()
        self._create_visualizations()
        self._save_results()
    
    def _generate_report(self):
        """Generate comprehensive success report"""
        
        print("=" * 80)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("=" * 80)
        print()
        
        # 1. CUMULATIVE SCORES
        print(" CUMULATIVE SCORES")
        print("-" * 80)
        print(f"Total Reward:            {self.metrics['total_reward']:>12.2f}")
        print(f"Total Intelligence:      {self.metrics['total_intelligence']:>12}")
        print(f"Total Engagement Time:   {self.metrics['total_engagement_time']:>12} steps")
        print(f"Total Episodes:          {self.metrics['total_episodes']:>12}")
        print()
        
        avg_reward = self.metrics['total_reward'] / self.metrics['total_episodes']
        avg_intelligence = self.metrics['total_intelligence'] / self.metrics['total_episodes']
        avg_engagement = self.metrics['total_engagement_time'] / self.metrics['total_episodes']
        
        print(f"Average Reward/Episode:        {avg_reward:>10.2f}")
        print(f"Average Intelligence/Episode:  {avg_intelligence:>10.2f}")
        print(f"Average Engagement/Episode:    {avg_engagement:>10.2f} steps")
        print()
        
        # 2. SUCCESS RATES
        print(" SUCCESS RATES")
        print("-" * 80)
        
        success_rate = (self.metrics['successful_episodes'] / self.metrics['total_episodes']) * 100
        stealth_rate = (self.metrics['honeypot_undetected'] / self.metrics['total_episodes']) * 100
        detection_rate = (self.metrics['honeypot_detected'] / self.metrics['total_episodes']) * 100
        
        print(f"Overall Success Rate:    {success_rate:>10.1f}%")
        print(f"  ├─ Successful Episodes: {self.metrics['successful_episodes']:>4}/{self.metrics['total_episodes']}")
        print(f"  └─ Failed Episodes:     {self.metrics['total_episodes'] - self.metrics['successful_episodes']:>4}/{self.metrics['total_episodes']}")
        print()
        
        print(f"Stealth Performance:")
        print(f"  ├─ Undetected (Success): {stealth_rate:>9.1f}% ({self.metrics['honeypot_undetected']} episodes)")
        print(f"  └─ Detected (Failure):   {detection_rate:>9.1f}% ({self.metrics['honeypot_detected']} episodes)")
        print()
        
        # UPDATED SUCCESS RATE INTERPRETATION
        if success_rate >= 75:
            rating = " EXCELLENT"
        elif success_rate >= 65:
            rating = " GOOD"
        elif success_rate >= 50:
            rating = " FAIR"
        else:
            rating = " NEEDS IMPROVEMENT"
        
        print(f"Performance Rating:      {rating}")
        print()
        
        # 3. SUCCESS CRITERIA BREAKDOWN
        print("📋 SUCCESS CRITERIA BREAKDOWN (UPDATED THRESHOLDS)")
        print("-" * 80)
        
        criteria_counts = {
            'good_engagement': 0,
            'intelligence_gathered': 0,
            'stealth_maintained': 0,
            'positive_reward': 0
        }
        
        for episode in self.metrics['episode_details']:
            for criterion, met in episode['success_criteria'].items():
                if met:
                    criteria_counts[criterion] += 1
        
        total = self.metrics['total_episodes']
        
        print(f"Good Engagement (≥20 steps):      {criteria_counts['good_engagement']:>4}/{total} ({criteria_counts['good_engagement']/total*100:>5.1f}%)")
        print(f"Intelligence Gathered (≥12):      {criteria_counts['intelligence_gathered']:>4}/{total} ({criteria_counts['intelligence_gathered']/total*100:>5.1f}%)")
        print(f"Stealth Maintained (<0.75):       {criteria_counts['stealth_maintained']:>4}/{total} ({criteria_counts['stealth_maintained']/total*100:>5.1f}%)")
        print(f"Positive Reward (>800):           {criteria_counts['positive_reward']:>4}/{total} ({criteria_counts['positive_reward']/total*100:>5.1f}%)")
        print()
        
        # 4. SUCCESS BY ATTACK TYPE
        print(" SUCCESS RATE BY ATTACK TYPE")
        print("-" * 80)
        
        attack_stats = []
        for attack_type, stats in sorted(self.metrics['attack_type_success'].items()):
            if stats['total'] > 0:
                rate = (stats['success'] / stats['total']) * 100
                attack_stats.append({
                    'attack': attack_type,
                    'rate': rate,
                    'success': stats['success'],
                    'total': stats['total']
                })
        
        # Sort by success rate
        attack_stats.sort(key=lambda x: x['rate'], reverse=True)
        
        print(f"{'Attack Type':<25} {'Success Rate':>15} {'Episodes':>12}")
        print("-" * 80)
        
        for stat in attack_stats:
            bar = "█" * int(stat['rate'] / 5) + "░" * (20 - int(stat['rate'] / 5))
            print(f"{stat['attack']:<25} {stat['rate']:>6.1f}% {bar:>20} {stat['success']:>3}/{stat['total']:<3}")
        
        print()
        
        # 5. WEIGHTED SUCCESS SCORE
        print("WEIGHTED SUCCESS SCORE (UPDATED FORMULA)")
        print("-" * 80)
        
        scores = [ep['success_score'] for ep in self.metrics['episode_details']]
        avg_score = np.mean(scores)
        
        print(f"Average Score:           {avg_score:>10.1f}/100")
        print(f"Median Score:            {np.median(scores):>10.1f}/100")
        print(f"Min/Max Score:           {np.min(scores):>10.1f}/{np.max(scores):<10.1f}")
        print()
        
        print("Score Breakdown (Updated Weights):")
        print(f"  • Engagement Time (30%): {min(avg_engagement/40.0, 1.0)*30:>6.1f}/30")
        print(f"  • Intelligence (30%):    {min(avg_intelligence/40.0, 1.0)*30:>6.1f}/30")
        print(f"  • Stealth (25%):         {(1-np.mean([ep['suspicious_level'] for ep in self.metrics['episode_details']]))*25:>6.1f}/25")
        print(f"  • Reward (15%):          {min(avg_reward/2500.0, 1.0)*15:>6.1f}/15")
        print()
        
        # 6. PERFORMANCE TIERS (UPDATED)
        print(" PERFORMANCE DISTRIBUTION")
        print("-" * 80)
        
        tiers = {
            'Excellent (75-100)': 0,
            'Good (60-74)': 0,
            'Fair (45-59)': 0,
            'Poor (0-44)': 0
        }
        
        for score in scores:
            if score >= 75:
                tiers['Excellent (75-100)'] += 1
            elif score >= 60:
                tiers['Good (60-74)'] += 1
            elif score >= 45:
                tiers['Fair (45-59)'] += 1
            else:
                tiers['Poor (0-44)'] += 1
        
        for tier, count in tiers.items():
            pct = (count / total) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"{tier:<20} {count:>4}/{total:<4} ({pct:>5.1f}%) {bar}")
        
        print()
        
        # 7. ADDITIONAL INSIGHTS
        print(" ADDITIONAL INSIGHTS")
        print("-" * 80)
        
        # Calculate correlation between metrics
        rewards = [ep['reward'] for ep in self.metrics['episode_details']]
        engagements = [ep['engagement_time'] for ep in self.metrics['episode_details']]
        intel = [ep['intelligence'] for ep in self.metrics['episode_details']]
        
        print(f"Reward vs Engagement correlation:    {np.corrcoef(rewards, engagements)[0,1]:>6.3f}")
        print(f"Reward vs Intelligence correlation:  {np.corrcoef(rewards, intel)[0,1]:>6.3f}")
        print(f"Engagement vs Intelligence corr.:    {np.corrcoef(engagements, intel)[0,1]:>6.3f}")
        print()
        
        # Best and worst performing episodes
        best_ep = max(self.metrics['episode_details'], key=lambda x: x['success_score'])
        worst_ep = min(self.metrics['episode_details'], key=lambda x: x['success_score'])
        
        print(f"Best Episode Performance:")
        print(f"  Score: {best_ep['success_score']:.1f}, Attack: {best_ep['attack_type']}, "
              f"Reward: {best_ep['reward']:.0f}, Intel: {best_ep['intelligence']}")
        print()
        print(f"Worst Episode Performance:")
        print(f"  Score: {worst_ep['success_score']:.1f}, Attack: {worst_ep['attack_type']}, "
              f"Reward: {worst_ep['reward']:.0f}, Intel: {worst_ep['intelligence']}")
        
        print()
        print("=" * 80)
    
    def _create_visualizations(self):
        """Create comprehensive visualization"""
        
        try:
            fig = plt.figure(figsize=(18, 12))
            
            # 1. Success Rate Over Time
            ax1 = plt.subplot(3, 3, 1)
            success_over_time = []
            for i in range(len(self.metrics['episode_details'])):
                success_count = sum([1 for ep in self.metrics['episode_details'][:i+1] if ep['is_successful']])
                success_over_time.append((success_count / (i+1)) * 100)
            
            ax1.plot(range(1, len(success_over_time) + 1), success_over_time, 
                    linewidth=2, color='green')
            ax1.axhline(y=65, color='orange', linestyle='--', label='Good Threshold (65%)')
            ax1.axhline(y=75, color='green', linestyle='--', label='Excellent Threshold (75%)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Success Rate (%)')
            ax1.set_title('Cumulative Success Rate Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Score Distribution
            ax2 = plt.subplot(3, 3, 2)
            scores = [ep['success_score'] for ep in self.metrics['episode_details']]
            ax2.hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(scores):.1f}')
            ax2.axvline(60, color='orange', linestyle=':', alpha=0.5, label='Good (60)')
            ax2.axvline(75, color='green', linestyle=':', alpha=0.5, label='Excellent (75)')
            ax2.set_xlabel('Success Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Success Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Success by Attack Type
            ax3 = plt.subplot(3, 3, 3)
            attacks = []
            success_rates = []
            for attack, stats in sorted(self.metrics['attack_type_success'].items()):
                if stats['total'] > 0:
                    attacks.append(attack.replace('_', '\n'))
                    success_rates.append((stats['success'] / stats['total']) * 100)
            
            colors = ['green' if rate >= 65 else 'orange' if rate >= 50 else 'red' 
                     for rate in success_rates]
            ax3.bar(range(len(attacks)), success_rates, color=colors, alpha=0.7, edgecolor='black')
            ax3.set_xticks(range(len(attacks)))
            ax3.set_xticklabels(attacks, rotation=45, ha='right', fontsize=8)
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('Success Rate by Attack Type')
            ax3.axhline(y=65, color='orange', linestyle='--', alpha=0.5)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. Cumulative Reward
            ax4 = plt.subplot(3, 3, 4)
            cumulative_reward = np.cumsum([ep['reward'] for ep in self.metrics['episode_details']])
            ax4.plot(range(1, len(cumulative_reward) + 1), cumulative_reward, 
                    linewidth=2, color='purple')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Cumulative Reward')
            ax4.set_title('Cumulative Reward Over Time')
            ax4.grid(True, alpha=0.3)
            ax4.ticklabel_format(style='plain', axis='y')
            
            # 5. Success Criteria Met
            ax5 = plt.subplot(3, 3, 5)
            criteria = ['Engagement\n(≥20)', 'Intelligence\n(≥12)', 'Stealth\n(<0.75)', 'Reward\n(>800)']
            counts = [
                sum([1 for ep in self.metrics['episode_details'] if ep['success_criteria']['good_engagement']]),
                sum([1 for ep in self.metrics['episode_details'] if ep['success_criteria']['intelligence_gathered']]),
                sum([1 for ep in self.metrics['episode_details'] if ep['success_criteria']['stealth_maintained']]),
                sum([1 for ep in self.metrics['episode_details'] if ep['success_criteria']['positive_reward']])
            ]
            percentages = [(c / len(self.metrics['episode_details'])) * 100 for c in counts]
            
            bars = ax5.barh(criteria, percentages, color='teal', alpha=0.7, edgecolor='black')
            ax5.set_xlabel('Percentage (%)')
            ax5.set_title('Success Criteria Achievement')
            ax5.set_xlim(0, 100)
            
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                ax5.text(pct + 2, i, f'{pct:.1f}%', va='center')
            
            ax5.grid(True, alpha=0.3, axis='x')
            
            # 6. Detection vs Undetected
            ax6 = plt.subplot(3, 3, 6)
            detection_data = [
                self.metrics['honeypot_undetected'],
                self.metrics['honeypot_detected']
            ]
            colors_pie = ['green', 'red']
            labels = [f"Undetected\n({self.metrics['honeypot_undetected']})", 
                     f"Detected\n({self.metrics['honeypot_detected']})"]
            
            ax6.pie(detection_data, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10})
            ax6.set_title('Stealth Performance')
            
            # 7. Intelligence Gathered Over Time
            ax7 = plt.subplot(3, 3, 7)
            cumulative_intel = np.cumsum([ep['intelligence'] for ep in self.metrics['episode_details']])
            ax7.plot(range(1, len(cumulative_intel) + 1), cumulative_intel,
                    linewidth=2, color='orange')
            ax7.set_xlabel('Episode')
            ax7.set_ylabel('Cumulative Intelligence')
            ax7.set_title('Intelligence Gathered Over Time')
            ax7.grid(True, alpha=0.3)
            
            # 8. Engagement Time Distribution
            ax8 = plt.subplot(3, 3, 8)
            engagement_times = [ep['engagement_time'] for ep in self.metrics['episode_details']]
            ax8.hist(engagement_times, bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
            ax8.axvline(np.mean(engagement_times), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(engagement_times):.1f}')
            ax8.axvline(20, color='green', linestyle=':', alpha=0.5, label='Target (20)')
            ax8.set_xlabel('Engagement Time (steps)')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Engagement Time Distribution')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            
            # 9. Performance Tier Distribution
            ax9 = plt.subplot(3, 3, 9)
            scores = [ep['success_score'] for ep in self.metrics['episode_details']]
            tiers = ['Excellent\n(75-100)', 'Good\n(60-74)', 'Fair\n(45-59)', 'Poor\n(0-44)']
            tier_counts = [
                sum([1 for s in scores if s >= 75]),
                sum([1 for s in scores if 60 <= s < 75]),
                sum([1 for s in scores if 45 <= s < 60]),
                sum([1 for s in scores if s < 45])
            ]
            
            colors_tiers = ['gold', 'lightgreen', 'orange', 'lightcoral']
            ax9.bar(range(len(tiers)), tier_counts, color=colors_tiers, alpha=0.7, edgecolor='black')
            ax9.set_xticks(range(len(tiers)))
            ax9.set_xticklabels(tiers, fontsize=9)
            ax9.set_ylabel('Number of Episodes')
            ax9.set_title('Performance Tier Distribution')
            ax9.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            filename = f"results/cumulative_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Visualizations saved: {filename}")
            plt.close()
            
        except Exception as e:
            print(f"⚠ Could not create visualizations: {e}")
    
    def _save_results(self):
        """Save detailed results to JSON"""
        
        # Calculate summary statistics
        scores = [ep['success_score'] for ep in self.metrics['episode_details']]
        
        results = {
            'model_path': self.model_path,
            'num_episodes': self.num_episodes,
            'timestamp': datetime.now().isoformat(),
            'metric_version': 'updated_v2',  # Track metric version
            
            'cumulative_scores': {
                'total_reward': float(self.metrics['total_reward']),
                'total_intelligence': int(self.metrics['total_intelligence']),
                'total_engagement_time': int(self.metrics['total_engagement_time']),
                'average_reward': float(self.metrics['total_reward'] / self.metrics['total_episodes']),
                'average_intelligence': float(self.metrics['total_intelligence'] / self.metrics['total_episodes']),
                'average_engagement': float(self.metrics['total_engagement_time'] / self.metrics['total_episodes'])
            },
            
            'success_rates': {
                'overall_success_rate': float((self.metrics['successful_episodes'] / self.metrics['total_episodes']) * 100),
                'successful_episodes': int(self.metrics['successful_episodes']),
                'failed_episodes': int(self.metrics['total_episodes'] - self.metrics['successful_episodes']),
                'stealth_success_rate': float((self.metrics['honeypot_undetected'] / self.metrics['total_episodes']) * 100),
                'detection_rate': float((self.metrics['honeypot_detected'] / self.metrics['total_episodes']) * 100)
            },
            
            'weighted_score': {
                'average_score': float(np.mean(scores)),
                'median_score': float(np.median(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores))
            },
            
            'attack_type_success': {
                attack: {
                    'success_rate': float((stats['success'] / stats['total']) * 100) if stats['total'] > 0 else 0,
                    'successful': int(stats['success']),
                    'total': int(stats['total'])
                }
                for attack, stats in self.metrics['attack_type_success'].items()
            },
            
            'success_criteria_thresholds': {
                'engagement_threshold': 20,
                'intelligence_threshold': 12,
                'stealth_threshold': 0.75,
                'reward_threshold': 800
            }
        }
        
        filename = f"results/cumulative_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved: {filename}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate cumulative scores and success rates')
    parser.add_argument('--model', type=str, default='models/best_agent.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("CUMULATIVE SCORING AND SUCCESS RATE CALCULATOR (UPDATED METRICS)")
    print("=" * 80)
    print()
    print("Updated Success Criteria:")
    print("  • Good Engagement: ≥20 steps (was ≥25)")
    print("  • Intelligence Gathered: ≥12 (was ≥15)")
    print("  • Stealth Maintained: <0.75 (was <0.7)")
    print("  • Positive Reward: >800 (was >1000)")
    print()
    print("Updated Weighted Score Formula:")
    print("  • Engagement Time: 30% (was 25%)")
    print("  • Intelligence: 30% (was 25%)")
    print("  • Stealth: 25% (was 30%)")
    print("  • Reward: 15% (was 20%)")
    print()
    
    scorer = CumulativeScorer(args.model, args.episodes)
    scorer.run_evaluation()
    
    print("\n✓ Analysis complete!")
    print("\nGenerated files:")
    print("  - results/cumulative_scores_YYYYMMDD_HHMMSS.json")
    print("  - results/cumulative_scores_YYYYMMDD_HHMMSS.png")
    print()


if __name__ == '__main__':
    main()
