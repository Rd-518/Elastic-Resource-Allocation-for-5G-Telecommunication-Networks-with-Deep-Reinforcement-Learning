import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple
import matplotlib.pyplot as plt

# Set device - use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NetworkSlicingEnv:
    """
    Network slicing environment that simulates a network provider handling slice requests
    for three types of services. Continuous operation mode with no terminal state.
    """
    def __init__(self, time_slot_length: int = 1):
        # Define resource limits: radio (Mbps), computing (CPU) and storage (GB) resources
        self.total_radio = 1000    # Radio resources
        self.total_computing = 20  # Computing resources
        self.total_storage = 10    # Storage resources
        self.time_slot_length = time_slot_length  # Time slot length
        
        # Service type parameters: [computing resource demand, storage resource demand, radio resource demand, 
        # request probability, unit reward, average service time, service time variance]
        self.service_types = [
            [2, 1, 100, 0.4, 1, 3, 1.0],   # Utilities service
            [4, 2, 150, 0.3, 2, 4, 1.5],   # Automotive service
            [6, 3, 200, 0.3, 4, 5, 2.0]    # Manufacturing service
        ]
        
        # Verify that request probabilities sum to 1
        assert abs(sum(st[3] for st in self.service_types) - 1.0) < 1e-6
        
        # Active slices queue
        self.active_slices = []
        
        # Action space (accept or reject)
        self.action_size = 2
        
        # State space size
        self.state_size = 7  # 3 normalized resources + 3 service type counts + 1 average remaining time
        
        self.reset()

    def reset(self):
        """Reset environment state"""
        self.active_slices = []
        self.current_step = 0
        return self._get_state()
    
    def _generate_request(self) -> Tuple[int, int]:
        """Generate a new slice request"""
        # Select service type based on given probabilities
        service_type = np.random.choice(3, p=[st[3] for st in self.service_types])
        
        # Generate service duration (normal distribution, minimum 1 time slot)
        mean = self.service_types[service_type][5]
        std = self.service_types[service_type][6]
        duration = max(1, int(np.random.normal(mean, std)))
        
        return service_type, duration
    
    def _get_available_resources(self) -> Tuple[int, int, int]:
        """Calculate currently available resources"""
        used_computing = sum(slice_req['resources'][0] for slice_req in self.active_slices)
        used_storage = sum(slice_req['resources'][1] for slice_req in self.active_slices)
        used_radio = sum(slice_req['resources'][2] for slice_req in self.active_slices)
        
        return (
            self.total_computing - used_computing,
            self.total_storage - used_storage,
            self.total_radio - used_radio
        )
    
    def _get_state(self) -> np.ndarray:
        """
        Build state representation, including:
          - Normalized available resources (3 values)
          - Active slice count for each service type (3 values)
          - Average remaining service time for active slices (1 value)
        Total dimensions: 7
        """
        available_computing, available_storage, available_radio = self._get_available_resources()
        norm_available = np.array([
            available_computing / self.total_computing,
            available_storage / self.total_storage,
            available_radio / self.total_radio
        ])
        
        # Count of active slices for each service type
        type_counts = np.zeros(3)
        total_remaining_time = 0
        
        for slice_req in self.active_slices:
            type_counts[slice_req['service_type']] += 1
            total_remaining_time += slice_req['remaining_time']
        
        # Normalize counts, assuming maximum of 50 requests
        norm_counts = type_counts / 50.0
        
        # Calculate average remaining time for all active slices (normalized, assuming maximum remaining time of 10)
        avg_remaining_time = total_remaining_time / len(self.active_slices) if self.active_slices else 0.0
        norm_avg_time = np.array([avg_remaining_time / 10.0])
        
        return np.concatenate((norm_available, norm_counts, norm_avg_time)).astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action for one time slot.
        Continuous system, done flag is always False.
        """
        total_reward = 0
        
        # 1. Reduce remaining time for all active slices
        for slice_req in self.active_slices:
            slice_req['remaining_time'] -= self.time_slot_length
        
        # 2. Remove completed slices and free resources
        self.active_slices = [slice_req for slice_req in self.active_slices if slice_req['remaining_time'] > 0]
        
        # 3. Generate new slice request
        service_type, duration = self._generate_request()
        computing, storage, radio, _, reward, _, _ = self.service_types[service_type]
        available_computing, available_storage, available_radio = self._get_available_resources()
        
        # 4. Try to allocate resources (if action is accept and resources are sufficient)
        if action == 1 and available_computing >= computing and available_storage >= storage and available_radio >= radio:
            self.active_slices.append({
                'service_type': service_type,
                'resources': (computing, storage, radio),
                'remaining_time': duration
            })
            total_reward += reward  # Immediate reward
        
        # 5. Update environment state
        self.current_step += 1
        done = False  # Continuous system, no terminal state
        
        # Additional information dictionary
        info = {
            'step': self.current_step,
            'active_slices': len(self.active_slices),
            'resources': self._get_available_resources()
        }
        
        return self._get_state(), total_reward, done, info


class QLearningAgent:
    """Q-Learning Agent"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Use dictionary to store Q-table for handling continuous state space
        self.q_table = {}
        
        # Learning parameters
        self.alpha = 0.01     # Learning rate
        self.gamma = 0.99      # Discount factor
        self.epsilon = 1.0     # Exploration rate
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        
        # State discretization parameters
        self.state_bins = 10   # Number of discretization bins per dimension
    
    def _discretize_state(self, state):
        """Discretize continuous state into fixed number of bins"""
        discretized = []
        for i, val in enumerate(state):
            # Discretize each state dimension
            bin_idx = min(int(val * self.state_bins), self.state_bins - 1)
            discretized.append(bin_idx)
        return tuple(discretized)
    
    def act(self, state):
        """Choose action: explore or exploit"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        # Discretize state
        state_key = self._discretize_state(state)
        
        # Initialize Q-values if state not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Choose action with highest Q-value
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table"""
        # Discretize current and next states
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)
        
        # Initialize Q-values if states not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Calculate Q-learning target
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        target_q = reward + (1 - done) * self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state_key][action] += self.alpha * (target_q - current_q)
    
    def update_parameters(self):
        """Update exploration rate and other parameters"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_q_table_size(self):
        """Return Q-table size for monitoring"""
        return len(self.q_table)


class DQN(nn.Module):
    """Deep Q Network"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_size)
        )
    
    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN Agent"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.epsilon = 1.0
        self.batch_size = 128
        self.lr = 0.1
        
        # Create policy network and target network
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        """Choose action based on current state"""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        """Learn from experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Calculate current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Calculate maximum Q-values for next states
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        
        # Calculate target Q-values
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Calculate loss and update network
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_parameters(self):
        """Update target network and decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.target_net.load_state_dict(self.policy_net.state_dict())


def calculate_sliding_window_avg(rewards, window_size=20):
    """Calculate sliding window average reward"""
    window_avg = []
    for i in range(len(rewards)):
        window_start = max(0, i - window_size + 1)
        window_avg.append(np.mean(rewards[window_start:i+1]))
    return window_avg


def save_data_for_matlab(filename, episodes, data_dict):
    """
    Save data to txt file for use in MATLAB
    Parameters:
        filename: output filename
        episodes: array of episode numbers
        data_dict: dictionary containing various data sequences, keys are data names
    """
    with open(filename, 'w') as f:
        # Write header row
        header = "Episode," + ",".join(data_dict.keys())
        f.write(header + "\n")
        
        # Write data rows
        for i, episode in enumerate(episodes):
            line = f"{episode}"
            for data_name in data_dict.keys():
                line += f",{data_dict[data_name][i]}"
            f.write(line + "\n")
    
    print(f"Data saved to {filename}")


def train_agent(agent_type='dqn', total_steps=30000, episode_steps=100, window_size=10, warmup_episodes=5):
    """
    Train a single agent, using sliding window average reward
    Parameters:
        agent_type: 'dqn' or 'q-learning'
        total_steps: total training steps
        episode_steps: steps per virtual episode
        window_size: sliding window size
        warmup_episodes: number of warmup episodes, not counted in performance evaluation
    """
    env = NetworkSlicingEnv(time_slot_length=1)
    state_size = env.state_size
    action_size = env.action_size
    
    # Create agent
    if agent_type.lower() == 'dqn':
        agent = DQNAgent(state_size=state_size, action_size=action_size)
    else:  # q-learning
        agent = QLearningAgent(state_size=state_size, action_size=action_size)
    
    # Record cumulative reward for each virtual episode
    episode_rewards = []
    epsilon_history = []  # Record exploration rate history
    
    # Initialize environment
    state = env.reset()
    cumulative_reward = 0
    
    # If warmup is needed, run a few episodes but don't record rewards
    if warmup_episodes > 0:
        print(f"Warming up environment for {warmup_episodes} episodes...")
        for step in range(warmup_episodes * episode_steps):
            action = agent.act(state)
            next_state, reward, _, _ = env.step(action)
            
            if agent_type.lower() == 'dqn':
                agent.remember(state, action, reward, next_state, False)
                agent.learn()
            else:  # q-learning
                agent.learn(state, action, reward, next_state, False)
            
            state = next_state
            
            if (step + 1) % episode_steps == 0:
                agent.update_parameters()
    
    # Training loop
    for step in range(total_steps):
        # Choose action
        action = agent.act(state)
        
        # Execute action
        next_state, reward, _, _ = env.step(action)  # done is always False
        
        # Store experience and learn
        if agent_type.lower() == 'dqn':
            agent.remember(state, action, reward, next_state, False)
            agent.learn()
        else:  # q-learning
            agent.learn(state, action, reward, next_state, False)
        
        # Accumulate reward
        cumulative_reward += reward
        
        # Update current state
        state = next_state
        
        # Every episode_steps steps as a "virtual episode"
        if (step + 1) % episode_steps == 0:
            episode_rewards.append(cumulative_reward)
            epsilon_history.append(agent.epsilon)  # Record current exploration rate
            
            if (step + 1) % (episode_steps * 5) == 0:
                if agent_type.lower() == 'q-learning':
                    q_table_size = agent.get_q_table_size()
                    print(f"Step {step+1}, Episode Reward: {cumulative_reward:.1f}, "
                          f"Epsilon: {agent.epsilon:.3f}, Q-table size: {q_table_size}")
                else:
                    print(f"Step {step+1}, Episode Reward: {cumulative_reward:.1f}, "
                          f"Epsilon: {agent.epsilon:.3f}")
            cumulative_reward = 0
            agent.update_parameters()
    
    # Calculate sliding window average reward
    window_avg = calculate_sliding_window_avg(episode_rewards, window_size)
    
    return agent, episode_rewards, window_avg, epsilon_history


def compare_agents(total_steps=50000, episode_steps=100, window_size=20, warmup_episodes=5):
    """Compare Q-Learning and DQN performance using sliding window average reward"""
    print("=== Starting comparison of Q-Learning and DQN ===")
    print(f"Training steps: {total_steps}, Steps per episode: {episode_steps}, Window size: {window_size}")
    print(f"Warmup episodes: {warmup_episodes}")
    
    # Train Q-Learning agent
    print("\nTraining Q-Learning agent...")
    ql_agent, ql_rewards, ql_window_avg, ql_epsilon = train_agent(
        'q-learning', total_steps, episode_steps, window_size, warmup_episodes)
    
    # Train DQN agent
    print("\nTraining DQN agent...")
    dqn_agent, dqn_rewards, dqn_window_avg, dqn_epsilon = train_agent(
        'dqn', total_steps, episode_steps, window_size, warmup_episodes)
    
    # Generate episode numbers
    episodes = np.arange(1, len(ql_rewards) + 1)
    
    # Save raw reward data to txt file (for MATLAB plotting)
    save_data_for_matlab("ql_dqn_raw_rewards.txt", episodes, {
        "QL_Reward": ql_rewards,
        "DQN_Reward": dqn_rewards
    })
    
    # Save sliding window average reward data to txt file (for MATLAB plotting)
    save_data_for_matlab("ql_dqn_window_avg.txt", episodes, {
        "QL_Window_Avg": ql_window_avg,
        "DQN_Window_Avg": dqn_window_avg
    })
    
    # Save exploration rate data to txt file (for MATLAB plotting)
    save_data_for_matlab("ql_dqn_epsilon.txt", episodes, {
        "QL_Epsilon": ql_epsilon,
        "DQN_Epsilon": dqn_epsilon
    })
    
    # Plot sliding window average reward comparison
    plt.figure(figsize=(12, 8))
    
    plt.plot(episodes, ql_window_avg, label='Q-Learning')
    plt.plot(episodes, dqn_window_avg, label='DQN')
    
    plt.xlabel('Episodes')
    plt.ylabel('Sliding Window Average Reward')
    plt.title(f'Q-Learning vs DQN for Network Slicing (Window Size: {window_size})')
    plt.legend()
    plt.grid(True)
    plt.savefig('ql_vs_dqn_sliding_window.png')
    plt.show()
    
    # Compare final performance (average of last 10 episodes)
    ql_final_perf = np.mean(ql_rewards[-10:])
    dqn_final_perf = np.mean(dqn_rewards[-10:])
    print(f"\nQ-Learning final performance (average of last 10 episodes): {ql_final_perf:.2f}")
    print(f"DQN final performance (average of last 10 episodes): {dqn_final_perf:.2f}")
    
    # Calculate stability (standard deviation of last 30 episodes)
    ql_stability = np.std(ql_rewards[-30:])
    dqn_stability = np.std(dqn_rewards[-30:])
    print(f"Q-Learning stability (standard deviation): {ql_stability:.2f}")
    print(f"DQN stability (standard deviation): {dqn_stability:.2f}")
    
    # Save final performance and stability data
    with open("ql_dqn_performance_metrics.txt", "w") as f:
        f.write("Metric,Q-Learning,DQN\n")
        f.write(f"Final_Performance,{ql_final_perf:.2f},{dqn_final_perf:.2f}\n")
        f.write(f"Stability,{ql_stability:.2f},{dqn_stability:.2f}\n")
        f.write(f"Final_Epsilon,{ql_epsilon[-1]:.4f},{dqn_epsilon[-1]:.4f}\n")
    
    return {
        'q-learning': (ql_agent, ql_rewards, ql_window_avg),
        'dqn': (dqn_agent, dqn_rewards, dqn_window_avg)
    }


if __name__ == "__main__":
    # Compare Q-Learning and DQN performance
    results = compare_agents(
        total_steps=400000,
        episode_steps=3000,
        window_size=45,
        warmup_episodes=2  # Add warmup phase to avoid initial peak
    )
    
    print("\nComparison complete. The following data files have been generated for MATLAB plotting:")
    print("- ql_dqn_raw_rewards.txt - Raw reward data")
    print("- ql_dqn_window_avg.txt - Sliding window average rewards")
    print("- ql_dqn_epsilon.txt - Exploration rate history")
    print("- ql_dqn_performance_metrics.txt - Performance metrics")