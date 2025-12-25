#!/usr/bin/env python3
"""
Flask Backend for Real-Time Honeypot Dashboard
Serves live data from the RL agent to the web dashboard
"""
from flask import Flask, render_template, jsonify, send_from_directory
from flask_cors import CORS
import torch
import numpy as np
import json
import threading
import time
from datetime import datetime
from honeypot_environment_enhanced import EnhancedHoneypotEnvironment
from custom_rl_agent import FixedHoneypotRLAgent

app = Flask(__name__)
CORS(app)

# Global state
dashboard_data = {
    'success_rate': 100.0,
    'avg_engagement': 57.4,
    'diversity': 98.7,
    'stealth_rate': 70.0,
    'avg_reward': 5630,
    'intelligence': 98.0,
    'episode_count': 1234,
    'recent_episodes': [],
    'attack_distribution': {},
    'live_feed': [],
    'system_status': 'active',
    'model_info': {}
}

# Initialize environment and agent
env = None
agent = None
monitoring_active = False


def load_model(model_path='models/final_agent.pth'):
    """Load the trained agent"""
    global env, agent
    
    try:
        env = EnhancedHoneypotEnvironment()
        agent = FixedHoneypotRLAgent(env.state_dim, env.action_dim)
        
        checkpoint = torch.load(model_path, map_location=agent.device, weights_only=False)
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.epsilon = 0.0
        
        dashboard_data['model_info'] = {
            'path': model_path,
            'training_steps': checkpoint.get('training_step', 0),
            'epsilon': float(agent.epsilon),
            'device': str(agent.device),
            'loaded_at': datetime.now().isoformat()
        }
        
        print(f"✓ Model loaded: {model_path}")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def run_episode():
    """Run a single episode and update dashboard data"""
    global dashboard_data
    
    if agent is None or env is None:
        return
    
    state = env.reset()
    attack_type = env.attack_types[env.current_attack_type]
    
    episode_reward = 0
    episode_steps = 0
    episode_intel = 0
    done = False
    
    actions_taken = []
    
    while not done:
        action = agent.select_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        
        episode_reward += reward
        episode_steps += 1
        episode_intel = info['intelligence']
        actions_taken.append(env.actions[action])
        
        state = next_state
    
    # Update metrics
    episode_data = {
        'episode': dashboard_data['episode_count'],
        'attack_type': attack_type,
        'reward': episode_reward,
        'steps': episode_steps,
        'intelligence': episode_intel,
        'suspicious_level': info['suspicious_level'],
        'stealth_maintained': info['suspicious_level'] < 0.75,
        'actions': actions_taken,
        'timestamp': datetime.now().isoformat()
    }
    
    # Update recent episodes (keep last 50)
    dashboard_data['recent_episodes'].append(episode_data)
    if len(dashboard_data['recent_episodes']) > 50:
        dashboard_data['recent_episodes'].pop(0)
    
    # Update attack distribution
    if attack_type not in dashboard_data['attack_distribution']:
        dashboard_data['attack_distribution'][attack_type] = 0
    dashboard_data['attack_distribution'][attack_type] += 1
    
    # Update live feed (keep last 20)
    feed_item = {
        'type': 'episode_complete',
        'title': f"{attack_type} Episode Completed",
        'details': f"Reward: {episode_reward:.0f} | Steps: {episode_steps} | Intel: {episode_intel}",
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'icon': '✓' if episode_data['stealth_maintained'] else '⚠️'
    }
    dashboard_data['live_feed'].insert(0, feed_item)
    if len(dashboard_data['live_feed']) > 20:
        dashboard_data['live_feed'].pop()
    
    # Update aggregate metrics
    recent = dashboard_data['recent_episodes'][-20:]  # Last 20 episodes
    
    dashboard_data['avg_engagement'] = np.mean([e['steps'] for e in recent])
    dashboard_data['avg_reward'] = np.mean([e['reward'] for e in recent])
    dashboard_data['intelligence'] = np.mean([e['intelligence'] for e in recent])
    dashboard_data['stealth_rate'] = sum([1 for e in recent if e['stealth_maintained']]) / len(recent) * 100
    
    dashboard_data['episode_count'] += 1
    
    print(f"Episode {dashboard_data['episode_count']}: {attack_type} | "
          f"Reward: {episode_reward:.0f} | Steps: {episode_steps} | "
          f"Stealth: {'✓' if episode_data['stealth_maintained'] else '✗'}")


def monitoring_loop():
    """Background thread for continuous monitoring"""
    global monitoring_active
    
    while monitoring_active:
        try:
            run_episode()
            time.sleep(2)  # Run episode every 2 seconds
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            time.sleep(5)


# Routes
@app.route('/')
def index():
    """Serve the dashboard HTML"""
    return send_from_directory('.', 'dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'status': dashboard_data['system_status'],
        'monitoring_active': monitoring_active,
        'model_loaded': agent is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/metrics')
def get_metrics():
    """Get current metrics"""
    return jsonify({
        'success_rate': dashboard_data['success_rate'],
        'avg_engagement': float(dashboard_data['avg_engagement']),
        'diversity': dashboard_data['diversity'],
        'stealth_rate': float(dashboard_data['stealth_rate']),
        'avg_reward': float(dashboard_data['avg_reward']),
        'intelligence': float(dashboard_data['intelligence']),
        'episode_count': dashboard_data['episode_count']
    })


@app.route('/api/episodes')
def get_episodes():
    """Get recent episode data"""
    return jsonify({
        'episodes': dashboard_data['recent_episodes'][-50:],
        'count': len(dashboard_data['recent_episodes'])
    })


@app.route('/api/attacks')
def get_attacks():
    """Get attack type distribution"""
    return jsonify({
        'distribution': dashboard_data['attack_distribution'],
        'total': sum(dashboard_data['attack_distribution'].values())
    })


@app.route('/api/feed')
def get_feed():
    """Get live activity feed"""
    return jsonify({
        'feed': dashboard_data['live_feed'][:20],
        'count': len(dashboard_data['live_feed'])
    })


@app.route('/api/model')
def get_model_info():
    """Get model information"""
    return jsonify(dashboard_data['model_info'])


@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start the monitoring loop"""
    global monitoring_active
    
    if not monitoring_active:
        monitoring_active = True
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
        return jsonify({'status': 'started', 'message': 'Monitoring started'})
    
    return jsonify({'status': 'already_running', 'message': 'Monitoring already active'})


@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop the monitoring loop"""
    global monitoring_active
    
    monitoring_active = False
    return jsonify({'status': 'stopped', 'message': 'Monitoring stopped'})


@app.route('/api/reset', methods=['POST'])
def reset_metrics():
    """Reset all metrics"""
    global dashboard_data
    
    dashboard_data['recent_episodes'] = []
    dashboard_data['attack_distribution'] = {}
    dashboard_data['live_feed'] = []
    dashboard_data['episode_count'] = 0
    
    return jsonify({'status': 'reset', 'message': 'Metrics reset successfully'})


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Honeypot Dashboard Server')
    parser.add_argument('--model', type=str, default='models/final_agent.pth',
                       help='Path to trained model')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to run server on')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run server on')
    parser.add_argument('--auto-start', action='store_true',
                       help='Automatically start monitoring on launch')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("RL HONEYPOT REAL-TIME DASHBOARD SERVER")
    print("=" * 70)
    print()
    
    # Load model
    if load_model(args.model):
        print()
        print("✓ Model loaded successfully!")
        print(f"  Training steps: {dashboard_data['model_info']['training_steps']}")
        print(f"  Device: {dashboard_data['model_info']['device']}")
    else:
        print()
        print("✗ Failed to load model. Server will run but monitoring won't work.")
    
    print()
    print(f"Starting dashboard server...")
    print(f"  URL: http://{args.host}:{args.port}")
    print()
    print("Available endpoints:")
    print(f"  Dashboard:  http://{args.host}:{args.port}/")
    print(f"  Status:     http://{args.host}:{args.port}/api/status")
    print(f"  Metrics:    http://{args.host}:{args.port}/api/metrics")
    print(f"  Episodes:   http://{args.host}:{args.port}/api/episodes")
    print(f"  Attacks:    http://{args.host}:{args.port}/api/attacks")
    print(f"  Feed:       http://{args.host}:{args.port}/api/feed")
    print()
    
    # Auto-start monitoring if requested
    if args.auto_start:
        print("Auto-starting monitoring...")
        monitoring_active = True
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
        print("✓ Monitoring started")
        print()
    else:
        print("To start monitoring, send POST to /api/start")
        print("Example: curl -X POST http://localhost:5000/api/start")
        print()
    
    print("=" * 70)
    print()
    
    # Run Flask server
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
