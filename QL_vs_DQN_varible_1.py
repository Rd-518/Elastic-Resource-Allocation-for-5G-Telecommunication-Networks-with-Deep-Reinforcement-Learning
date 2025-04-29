import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple

# Set device - use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EnhancedNetworkSlicingEnv:
    """
    Enhanced Network slicing environment with increased complexity:
    - Dynamic resource fluctuation
    - More service types
    - User priority system
    - Time-related request patterns
    - Failure simulation
    - Security requirements (Levels 1-3, where 1 is highest)
    - Reliability requirements (Levels 1-3, where 1 is highest)
    """
    def __init__(self, time_slot_length: int = 1, type1_reward: float = 1.0):
        # Base resource limits: radio (Mbps), computing (CPU) and storage (GB) resources
        self.base_total_radio = 400    # Base radio resources
        self.base_total_computing = 10  # Base computing resources
        self.base_total_storage = 8    # Base storage resources
        
        # Current available resources (will fluctuate over time)
        self.total_radio = self.base_total_radio
        self.total_computing = self.base_total_computing
        self.total_storage = self.base_total_storage
        
        self.time_slot_length = time_slot_length  # Time slot length
        self.current_hour = 0  # Current hour (for simulating different traffic patterns)
        
        # Service type parameters: [computing demand, storage demand, radio demand, 
        # request probability, unit reward, average service time, service time variance, priority weight,
        # security level, reliability level, security overhead factor, reliability overhead factor]
        # Security & Reliability levels: 1=Highest, 2=Average, 3=Low
        self.service_types = [
            [2, 1, 100, 0.25, type1_reward, 3, 1.0, 1, 3, 3, 1.0, 1.0],     # Utilities service (low security & reliability)
            [4, 2, 150, 0.20, 2, 4, 1.5, 2, 2, 2, 1.3, 1.2],     # Automotive service (average security & reliability)
            [6, 3, 200, 0.15, 4, 5, 2.0, 3, 1, 2, 1.5, 1.2],     # Manufacturing service (high security, average reliability)
            [1, 0.5, 80, 0.15, 0.5, 2, 0.8, 1, 3, 3, 1.0, 1.0],  # IoT service (low security & reliability)
            [5, 4, 180, 0.15, 3, 6, 2.5, 2, 2, 1, 1.3, 1.5],     # AR service (average security, high reliability)
            [8, 5, 250, 0.10, 5, 7, 3.0, 3, 1, 1, 1.5, 1.5],     # Telemedicine service (high security & reliability)
        ]
        
        # Resource fluctuation parameters
        self.resource_fluctuation_period = 24  # Resource fluctuation period (hours)
        self.resource_fluctuation_amplitude = 0.2  # Resource fluctuation amplitude (percentage of base value)
        
        # Peak hours settings (e.g., 8-10 AM and 5-7 PM)
        self.peak_hours = [(8, 10), (17, 19)]
        self.peak_factor = 2.0  # Multiplier for request probability during peak hours
        
        # Failure simulation parameters
        self.failure_probability = 0.005  # Probability of failure per time slot
        self.failure_duration = 5  # Duration of failure (time slots)
        self.failure_severity = 0.5  # Severity of failure (percentage of resource reduction)
        self.current_failure = None  # Current failure status
        
        # Active slices queue
        self.active_slices = []
        
        # History records
        self.history = {
            'accepted_requests': [0] * 6,  # Number of accepted requests per service type
            'rejected_requests': [0] * 6,  # Number of rejected requests per service type
            'resource_utilization': [],  # Resource utilization history
            'revenue': 0,  # Total revenue
            'security_levels': [0, 0, 0],  # Count of requests handled by security level (1, 2, 3)
            'reliability_levels': [0, 0, 0],  # Count of requests handled by reliability level (1, 2, 3)
            'service_rewards': [0] * 6,  # Total rewards per service type
            'service_request_counts': [0] * 6  # Request counts per service type for calculating average
        }
        
        # Action space (accept or reject)
        self.action_size = 2
        
        # State space size
        # 3 normalized resources + 6 service type counts + 1 average remaining time + 1 current hour + 
        # 1 failure indicator + 2 for average security and reliability levels in active slices
        self.state_size = 14
        
        self.reset()

    def reset(self):
        """Reset environment state"""
        self.active_slices = []
        self.current_step = 0
        self.current_hour = 0
        self.current_failure = None
        
        # Reset history
        self.history = {
            'accepted_requests': [0] * 6,
            'rejected_requests': [0] * 6,
            'resource_utilization': [],
            'revenue': 0,
            'security_levels': [0, 0, 0],
            'reliability_levels': [0, 0, 0],
            'service_rewards': [0] * 6,
            'service_request_counts': [0] * 6
        }
        
        # Initialize dynamic resources
        self._update_resources()
        
        return self._get_state()
    
    def _is_peak_hour(self):
        """Check if current hour is a peak hour"""
        for start, end in self.peak_hours:
            if start <= self.current_hour < end:
                return True
        return False
    
    def _update_resources(self):
        """Update dynamic resource levels"""
        # Time-based resource fluctuation
        time_factor = np.sin(2 * np.pi * self.current_hour / self.resource_fluctuation_period)
        fluctuation = 1.0 + self.resource_fluctuation_amplitude * time_factor
        
        # Apply fluctuation factor
        self.total_radio = self.base_total_radio * fluctuation
        self.total_computing = self.base_total_computing * fluctuation
        self.total_storage = self.base_total_storage * fluctuation
        
        # If there's a failure, reduce resources
        if self.current_failure:
            self.total_radio *= (1 - self.failure_severity)
            self.total_computing *= (1 - self.failure_severity)
            self.total_storage *= (1 - self.failure_severity)
            
            # Update failure duration
            self.current_failure -= 1
            if self.current_failure <= 0:
                self.current_failure = None
    
    def _check_for_failure(self):
        """Check if a failure occurs"""
        if self.current_failure is None and random.random() < self.failure_probability:
            self.current_failure = self.failure_duration
            return True
        return False
    
    def _generate_request(self) -> Tuple[int, int, int]:
        """Generate a new slice request, returns service type, duration, and user priority"""
        # Adjust request probabilities to reflect peak hours
        probs = [st[3] for st in self.service_types]
        if self._is_peak_hour():
            # Increase request probability for high-priority services during peak hours
            for i, service in enumerate(self.service_types):
                priority = service[7]  # Get service priority
                if priority > 1:  # Medium and high priority services
                    probs[i] *= self.peak_factor
            
            # Renormalize probabilities
            probs = [p/sum(probs) for p in probs]
        
        # Select service type based on given probabilities
        service_type = np.random.choice(len(self.service_types), p=probs)
        
        # Generate service duration (normal distribution, minimum 1 time slot)
        mean = self.service_types[service_type][5]
        std = self.service_types[service_type][6]
        duration = max(1, int(np.random.normal(mean, std)))
        
        # Random user priority (1-3, higher number means higher priority)
        user_priority = random.randint(1, 3)
        
        return service_type, duration, user_priority
    
    def _calculate_resource_requirements(self, service_type):
        """Calculate resource requirements with security and reliability overhead"""
        service_params = self.service_types[service_type]
        base_computing = service_params[0]
        base_storage = service_params[1]
        base_radio = service_params[2]
        
        # Get security and reliability parameters
        security_level = service_params[8]      # 1 (highest) to 3 (lowest)
        reliability_level = service_params[9]   # 1 (highest) to 3 (lowest)
        security_overhead = service_params[10]  # Overhead factor
        reliability_overhead = service_params[11]  # Overhead factor
        
        # Calculate security and reliability overhead
        security_impact = (4 - security_level) * (security_overhead - 1.0) / 3.0  # Normalize impact
        reliability_impact = (4 - reliability_level) * (reliability_overhead - 1.0) / 3.0
        
        # Apply overhead to resource requirements
        computing = base_computing * (1 + security_impact * 0.2 + reliability_impact * 0.1)
        storage = base_storage * (1 + security_impact * 0.3 + reliability_impact * 0.05)
        radio = base_radio * (1 + security_impact * 0.05 + reliability_impact * 0.2)
        
        return computing, storage, radio
    
    def _get_available_resources(self) -> Tuple[float, float, float]:
        """Calculate currently available resources"""
        used_computing = sum(slice_req['resources'][0] for slice_req in self.active_slices)
        used_storage = sum(slice_req['resources'][1] for slice_req in self.active_slices)
        used_radio = sum(slice_req['resources'][2] for slice_req in self.active_slices)
        
        return (
            self.total_computing - used_computing,
            self.total_storage - used_storage,
            self.total_radio - used_radio
        )
    
    def _calculate_resource_utilization(self) -> Tuple[float, float, float]:
        """Calculate resource utilization"""
        available_computing, available_storage, available_radio = self._get_available_resources()
        
        computing_utilization = 1 - (available_computing / self.total_computing)
        storage_utilization = 1 - (available_storage / self.total_storage)
        radio_utilization = 1 - (available_radio / self.total_radio)
        
        return computing_utilization, storage_utilization, radio_utilization
    
    def _get_state(self) -> np.ndarray:
        """
        Build state representation, including:
          - Normalized available resources (3 values)
          - Active slice count for each service type (6 values)
          - Average remaining service time (1 value)
          - Current hour (1 value, normalized to [0,1])
          - Failure indicator (1 value, 0 or 1)
          - Average security level of active slices (1 value, normalized)
          - Average reliability level of active slices (1 value, normalized)
        Total dimensions: 14
        """
        available_computing, available_storage, available_radio = self._get_available_resources()
        norm_available = np.array([
            available_computing / self.total_computing,
            available_storage / self.total_storage,
            available_radio / self.total_radio
        ])
        
        # Count of active slices for each service type
        type_counts = np.zeros(len(self.service_types))
        total_remaining_time = 0
        total_security_level = 0
        total_reliability_level = 0
        
        for slice_req in self.active_slices:
            service_type = slice_req['service_type']
            type_counts[service_type] += 1
            total_remaining_time += slice_req['remaining_time']
            total_security_level += self.service_types[service_type][8]
            total_reliability_level += self.service_types[service_type][9]
        
        # Normalize counts, assuming maximum of 50 requests per type
        norm_counts = type_counts / 50.0
        
        # Calculate average remaining time for all active slices (normalized, assuming maximum remaining time of 10)
        avg_remaining_time = total_remaining_time / len(self.active_slices) if self.active_slices else 0.0
        norm_avg_time = np.array([avg_remaining_time / 10.0])
        
        # Current hour (normalized to [0,1])
        norm_hour = np.array([self.current_hour / 24.0])
        
        # Failure indicator
        failure_indicator = np.array([1.0 if self.current_failure else 0.0])
        
        # Average security and reliability levels (normalized)
        # Scale so 1 (highest) becomes 1.0 and 3 (lowest) becomes 0.0
        avg_security = total_security_level / len(self.active_slices) if self.active_slices else 2.0
        avg_reliability = total_reliability_level / len(self.active_slices) if self.active_slices else 2.0
        
        norm_security = np.array([(3 - avg_security) / 2.0])  # Normalized to [0,1] where 1 means highest security
        norm_reliability = np.array([(3 - avg_reliability) / 2.0])  # Normalized to [0,1] where 1 means highest reliability
        
        return np.concatenate((
            norm_available, 
            norm_counts, 
            norm_avg_time, 
            norm_hour, 
            failure_indicator, 
            norm_security, 
            norm_reliability
        )).astype(np.float32)
    
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
        completed_slices = [slice_req for slice_req in self.active_slices if slice_req['remaining_time'] <= 0]
        self.active_slices = [slice_req for slice_req in self.active_slices if slice_req['remaining_time'] > 0]
        
        # 3. Update time and check for failure
        self.current_hour = (self.current_hour + self.time_slot_length) % 24
        self._check_for_failure()
        self._update_resources()
        
        # 4. Generate new slice request
        service_type, duration, user_priority = self._generate_request()
        service_params = self.service_types[service_type]
        reward_per_time = service_params[4]
        priority_weight = service_params[7]
        
        # Track request for this service type
        self.history['service_request_counts'][service_type] += 1
        
        # Get security and reliability levels
        security_level = service_params[8]
        reliability_level = service_params[9]
        
        # Calculate resource requirements with security and reliability overhead
        computing, storage, radio = self._calculate_resource_requirements(service_type)
        
        # Get available resources
        available_computing, available_storage, available_radio = self._get_available_resources()
        
        # 5. Try to allocate resources (if action is accept and resources are sufficient)
        request_accepted = False
        
        if action == 1 and available_computing >= computing and available_storage >= storage and available_radio >= radio:
            # Calculate combined priority (service priority * user priority)
            combined_priority = priority_weight * user_priority
            
            # Additional reward based on security and reliability requirements
            sec_rel_bonus = (4 - security_level) * 0.2 + (4 - reliability_level) * 0.2
            
            # Calculate total reward for this request (base reward * duration * combined priority * sec/rel bonus)
            request_reward = reward_per_time * duration * combined_priority * (1 + sec_rel_bonus)
            
            self.active_slices.append({
                'service_type': service_type,
                'resources': (computing, storage, radio),
                'remaining_time': duration,
                'priority': combined_priority,
                'security': security_level,
                'reliability': reliability_level,
                'reward': request_reward
            })
            
            # Update history
            self.history['accepted_requests'][service_type] += 1
            self.history['revenue'] += request_reward
            self.history['security_levels'][security_level-1] += 1
            self.history['reliability_levels'][reliability_level-1] += 1
            
            # Track rewards by service type
            self.history['service_rewards'][service_type] += request_reward
            
            total_reward += request_reward  # Immediate reward
            request_accepted = True
        else:
            # Update rejected request history
            self.history['rejected_requests'][service_type] += 1
            
            # Negative reward for rejecting high-priority requests or requests with high sec/rel requirements
            rejection_penalty = priority_weight * user_priority
            
            # Additional penalty for rejecting high security/reliability requests
            if security_level == 1 or reliability_level == 1:  # Highest level
                rejection_penalty *= 1.5
            
            if priority_weight > 1:
                total_reward -= rejection_penalty
        
        # 6. Update environment state
        self.current_step += 1
        
        # Record resource utilization
        utilization = self._calculate_resource_utilization()
        self.history['resource_utilization'].append(utilization)
        
        # Continuous system, no terminal state
        done = False
        
        # Additional information dictionary
        info = {
            'step': self.current_step,
            'hour': self.current_hour,
            'active_slices': len(self.active_slices),
            'resources': self._get_available_resources(),
            'utilization': utilization,
            'request': {
                'type': service_type,
                'duration': duration,
                'user_priority': user_priority,
                'security_level': security_level,
                'reliability_level': reliability_level,
                'accepted': request_accepted
            },
            'failure': self.current_failure is not None,
            'history': self.history
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
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
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
        self.epsilon_decay = 0.93
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.epsilon = 1.0
        self.batch_size = 512
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
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
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
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Calculate current Q-values
        current_q = self.policy_net(states).gather(1, actions)
        
        # Calculate maximum Q-values for next states
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # Calculate target Q-values
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Calculate loss and update network
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_parameters(self):
        """Update target network and decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.target_net.load_state_dict(self.policy_net.state_dict())


class GreedyAgent:
    """Greedy Agent for network slicing"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # No learning parameters needed for greedy agent
        # Could add some statistics tracking here if needed
        
    def act(self, state, env=None):
        """Always choose accept (1) if resources are available"""
        # Greedy always returns 1 (accept) and relies on the environment
        # to reject if resources are insufficient
        return 1
    
    def remember(self, *args):
        """No memory needed for greedy agent"""
        pass
    
    def learn(self, *args):
        """No learning needed for greedy agent"""
        pass
    
    def update_parameters(self):
        """No parameters to update"""
        pass


def calculate_cumulative_avg(rewards):
    """计算累计平均奖励"""
    cumulative_sum = np.cumsum(rewards)
    indices = np.arange(1, len(rewards) + 1)
    cumulative_avg = cumulative_sum / indices
    return cumulative_avg
def train_agent(agent_type='dqn', total_steps=20000, episode_steps=100, window_size=10, warmup_episodes=5, type1_reward=1.0):
    """
    Train a single agent, using sliding window average reward
    Parameters:
        agent_type: 'dqn', 'q-learning', or 'greedy'
        total_steps: total training steps
        episode_steps: steps per virtual episode
        window_size: sliding window size
        warmup_episodes: number of warmup episodes, not counted in performance evaluation
        type1_reward: reward value for type1 (Utilities service)
    """
    env = EnhancedNetworkSlicingEnv(time_slot_length=1, type1_reward=type1_reward)
    
    state_size = env.state_size
    action_size = env.action_size
    
    # Create agent
    if agent_type.lower() == 'dqn':
        agent = DQNAgent(state_size=state_size, action_size=action_size)
    elif agent_type.lower() == 'greedy':
        agent = GreedyAgent(state_size=state_size, action_size=action_size)
    else:  # q-learning
        agent = QLearningAgent(state_size=state_size, action_size=action_size)
    
    # Record cumulative reward for each virtual episode
    episode_rewards = []
    epsilon_history = []  # Record exploration rate history (not used for greedy)
    
    # 保存历史记录
    final_history = None
    
    # Initialize environment
    state = env.reset()
    cumulative_reward = 0
    
    # If warmup is needed, run a few episodes but don't record rewards
    # Skip warmup for greedy agent since it doesn't learn
    if warmup_episodes > 0 and agent_type.lower() != 'greedy':
        print(f"Warming up environment for {warmup_episodes} episodes...")
        for step in range(warmup_episodes * episode_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            if agent_type.lower() == 'dqn':
                agent.remember(state, action, reward, next_state, done)
                agent.learn()
            elif agent_type.lower() == 'q-learning':
                agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            
            if (step + 1) % episode_steps == 0:
                agent.update_parameters()
    
    # Training loop
    for step in range(total_steps):
        # Choose action
        action = agent.act(state)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Store experience and learn (skip for greedy)
        if agent_type.lower() == 'dqn':
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
        elif agent_type.lower() == 'q-learning':
            agent.learn(state, action, reward, next_state, done)
        
        # Accumulate reward
        cumulative_reward += reward
        
        # Update current state
        state = next_state
        
        # Every episode_steps steps as a "virtual episode"
        if (step + 1) % episode_steps == 0:
            episode_rewards.append(cumulative_reward)
            
            if agent_type.lower() != 'greedy':
                epsilon_history.append(agent.epsilon)  # Record current exploration rate
            
            if (step + 1) % (episode_steps * 5) == 0:
                if agent_type.lower() == 'q-learning':
                    q_table_size = agent.get_q_table_size()
                    print(f"Step {step+1}, Episode Reward: {cumulative_reward:.1f}, "
                          f"Epsilon: {agent.epsilon:.3f}, Q-table size: {q_table_size}")
                elif agent_type.lower() == 'dqn':
                    print(f"Step {step+1}, Episode Reward: {cumulative_reward:.1f}, "
                          f"Epsilon: {agent.epsilon:.3f}")
                else:  # greedy
                    print(f"Step {step+1}, Episode Reward: {cumulative_reward:.1f}")
            
            cumulative_reward = 0
            
            if agent_type.lower() != 'greedy':
                agent.update_parameters()
            
            # 保存最后一个episode的历史记录
            if step + 1 == total_steps:
                final_history = info['history']
    
    # 计算累计平均奖励
    cumulative_avg = calculate_cumulative_avg(episode_rewards)
    
    return episode_rewards, cumulative_avg, final_history


def save_dynamic_reward_performance(filename, type1_rewards, ql_perf, dqn_perf, greedy_perf=None):
    """
    保存综合性能数据到文本文件
    参数:
        filename: 输出文件名
        type1_rewards: 奖励值列表
        ql_perf: Q-Learning的性能值
        dqn_perf: DQN的性能值
        greedy_perf: Greedy的性能值 (可选)
    """
    with open(filename, 'w') as f:
        # 写入表头
        if greedy_perf is not None:
            f.write("Type1_Reward,Q-Learning_Performance,DQN_Performance,Greedy_Performance\n")
        else:
            f.write("Type1_Reward,Q-Learning_Performance,DQN_Performance\n")
        
        # 写入每个奖励值的性能数据
        for i, reward in enumerate(type1_rewards):
            if greedy_perf is not None:
                f.write(f"{reward},{ql_perf[i]:.2f},{dqn_perf[i]:.2f},{greedy_perf[i]:.2f}\n")
            else:
                f.write(f"{reward},{ql_perf[i]:.2f},{dqn_perf[i]:.2f}\n")
    
    print(f"综合性能数据已保存到 {filename}")


def save_service_acceptance_data(filename, type1_rewards, service_acceptance_data):
    """
    保存每种服务类型的总接受量数据
    参数:
        filename: 输出文件名
        type1_rewards: 奖励值列表
        service_acceptance_data: 包含服务接受量数据的字典
    """
    with open(filename, 'w') as f:
        # 写入表头
        service_types = ['Utilities', 'Automotive', 'Manufacturing', 'IoT', 'AR', 'Telemedicine']
        header = "Reward,Algorithm," + ",".join(service_types)
        f.write(header + "\n")
        
        # 写入每个奖励值的数据
        for reward in type1_rewards:
            for alg in service_acceptance_data[reward].keys():  # 支持动态算法列表
                # 获取该算法和奖励值下的服务接受量数据
                acceptance = service_acceptance_data[reward][alg]
                
                # 写入数据行
                line = f"{reward},{alg}," + ",".join(str(val) for val in acceptance)
                f.write(line + "\n")
    
    print(f"服务类型接受量数据已保存到 {filename}")


def dynamic_type1_reward_experiment(total_steps_per_reward=20000, episode_steps=500, window_size=20, warmup_episodes=5):
    """
    Run experiment with dynamic Type1 reward from 1 to 9
    Parameters:
        total_steps_per_reward: total training steps for each reward value
        episode_steps: steps per virtual episode
        window_size: sliding window size
        warmup_episodes: number of warmup episodes per reward value
    """
    print("=== Starting Dynamic Type1 Reward Experiment with Enhanced Environment ===")
    print(f"Training steps per reward: {total_steps_per_reward}")
    print(f"Steps per episode: {episode_steps}")
    print(f"Window size: {window_size}")
    
    # Define reward values to test
    type1_rewards = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # Store final performance for each reward value
    ql_final_perf = []
    dqn_final_perf = []
    greedy_final_perf = []
    
    # 存储服务类型接受量数据
    service_acceptance_data = {}
    
    # Train for each reward value
    for reward in type1_rewards:
        print(f"\n=== Training with Type1 Reward = {reward} ===")
        
        # 为当前奖励值初始化接受量数据存储
        service_acceptance_data[reward] = {
            'QL': [0] * 6,
            'DQN': [0] * 6,
            'Greedy': [0] * 6
        }
        
        # Train Q-Learning agent
        print(f"\nTraining Q-Learning agent with Type1 Reward = {reward}...")
        ql_rewards, ql_cumulative_avg, ql_history = train_agent(
            'q-learning', total_steps_per_reward, episode_steps, window_size, warmup_episodes, reward)
        
        # 保存Q-Learning的服务接受量数据
        service_acceptance_data[reward]['QL'] = ql_history['accepted_requests']
        
        # Train DQN agent
        print(f"\nTraining DQN agent with Type1 Reward = {reward}...")
        dqn_rewards, dqn_cumulative_avg, dqn_history = train_agent(
            'dqn', total_steps_per_reward, episode_steps, window_size, warmup_episodes, reward)
        
        # 保存DQN的服务接受量数据
        service_acceptance_data[reward]['DQN'] = dqn_history['accepted_requests']
        
        # Train Greedy agent
        print(f"\nTraining Greedy agent with Type1 Reward = {reward}...")
        greedy_rewards, greedy_cumulative_avg, greedy_history = train_agent(
            'greedy', total_steps_per_reward, episode_steps, window_size, 0, reward)  # Greedy不需要热身
        
        # 保存Greedy的服务接受量数据
        service_acceptance_data[reward]['Greedy'] = greedy_history['accepted_requests']
        
        # 使用最终累计平均作为最终性能指标
        ql_final = ql_cumulative_avg[-1]  # 最终累计平均
        dqn_final = dqn_cumulative_avg[-1]  # 最终累计平均
        greedy_final = greedy_cumulative_avg[-1]  # 最终累计平均
        
        ql_final_perf.append(ql_final)
        dqn_final_perf.append(dqn_final)
        greedy_final_perf.append(greedy_final)
        
        print(f"\nQ-Learning 最终累计平均性能 (reward={reward}): {ql_final:.2f}")
        print(f"DQN 最终累计平均性能 (reward={reward}): {dqn_final:.2f}")
        print(f"Greedy 最终累计平均性能 (reward={reward}): {greedy_final:.2f}")
    
    # 保存性能数据到文本文件
    save_dynamic_reward_performance(
        "enhanced_dynamic_reward_performance.txt", 
        type1_rewards, 
        ql_final_perf, 
        dqn_final_perf,
        greedy_final_perf
    )
    
    # 保存服务类型接受量数据
    save_service_acceptance_data(
        "enhanced_service_acceptance.txt", 
        type1_rewards, 
        service_acceptance_data
    )
    
    print("\n增强环境实验完成！所有数据保存在以下文件中:")
    print("- enhanced_dynamic_reward_performance.txt - 不同奖励值下各算法的性能数据")
    print("- enhanced_service_acceptance.txt - 不同奖励值下各服务类型的接受量数据")
    
    return ql_final_perf, dqn_final_perf, greedy_final_perf, service_acceptance_data


if __name__ == "__main__":
    # Run dynamic reward experiment with enhanced environment
    ql_perf, dqn_perf, greedy_perf, service_acceptance = dynamic_type1_reward_experiment(
        total_steps_per_reward=50000,
        window_size=10,
        warmup_episodes=2
    )
