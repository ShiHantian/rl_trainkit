import torch
import numpy as np


class Buffer:
    """Base buffer class for storing transitions.

    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def push_one_step(self, state, action, reward, next_state, done, value, log_prob):
        """Push one step of experience to buffer.

        Args:
            state (np.ndarray): Current state
            action (np.ndarray): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Episode termination flag
            value (float): State value
            log_prob (float): Action log probability

        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def concat(self, other_buffer):
        """Concatenate another buffer to this one.

        Args:
            other_buffer (Buffer): Buffer to concatenate

        """
        self.states.extend(other_buffer.states)
        self.actions.extend(other_buffer.actions)
        self.rewards.extend(other_buffer.rewards)
        self.next_states.extend(other_buffer.next_states)
        self.dones.extend(other_buffer.dones)
        self.values.extend(other_buffer.values)
        self.log_probs.extend(other_buffer.log_probs)

    def retrieve_all(self):
        """Retrieve all data from buffer.

        Returns:
            data (dict): Dictionary containing all buffer data

        """
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'next_states': np.array(self.next_states),
            'dones': np.array(self.dones),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs)
        }

    def clear(self):
        """Clear all data from buffer.

        """
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()

    def __len__(self):
        return len(self.states)


class TrajectoryBuffer(Buffer):
    """Buffer for storing and processing trajectories.

    Args:
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter

    """

    def __init__(self, gamma=0.99, gae_lambda=0.95):
        super().__init__()
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.returns = []
        self.advantages = []

    def compute_returns_and_advantage(self, last_value=0):
        """Compute returns and advantages using GAE.

        Args:
            last_value (float): Value of the last state

        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        # Add last value for computation
        values = np.append(values, last_value)

        # Compute GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]

        self.returns = returns.tolist()
        self.advantages = advantages.tolist()

    def retrieve_all(self):
        """Retrieve all data including returns and advantages.

        Returns:
            data (dict): Dictionary containing all buffer data

        """
        data = super().retrieve_all()
        data['returns'] = np.array(self.returns)
        data['advantages'] = np.array(self.advantages)
        return data


class RolloutBuffer(Buffer):
    """Buffer for storing rollout data and creating batches.

    Args:
        threshold_rollout_length (int): Minimum rollout length
        batch_size (int): Batch size for training
        batch_num (int): Number of batches (if None, use batch_size)

    """

    def __init__(self, threshold_rollout_length=2048, batch_size=64, batch_num=None):
        super().__init__()
        self.threshold_rollout_length = threshold_rollout_length
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.returns = []
        self.advantages = []

    def concat(self, trajectory_buffer):
        """Concatenate trajectory buffer to rollout buffer.

        Args:
            trajectory_buffer (TrajectoryBuffer): Trajectory buffer to concatenate

        """
        super().concat(trajectory_buffer)
        self.returns.extend(trajectory_buffer.returns)
        self.advantages.extend(trajectory_buffer.advantages)

    def get_batches(self):
        """Generate batches for training.

        Yields:
            batch (dict): Batch of data

        """
        data = self.retrieve_all()
        n_samples = len(self.states)

        # Normalize advantages
        data['advantages'] = (data['advantages'] - data['advantages'].mean()) / (data['advantages'].std() + 1e-8)

        # Determine batch configuration
        if self.batch_num is not None:
            batch_size = n_samples // self.batch_num
        else:
            batch_size = self.batch_size

        # Create random indices
        indices = np.random.permutation(n_samples)

        # Yield batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch = {
                'states': torch.FloatTensor(data['states'][batch_indices]),
                'actions': torch.FloatTensor(data['actions'][batch_indices]),
                'log_probs': torch.FloatTensor(data['log_probs'][batch_indices]),
                'returns': torch.FloatTensor(data['returns'][batch_indices]),
                'advantages': torch.FloatTensor(data['advantages'][batch_indices]),
                'values': torch.FloatTensor(data['values'][batch_indices])
            }

            yield batch

    def retrieve_all(self):
        """Retrieve all data including returns and advantages.

        Returns:
            data (dict): Dictionary containing all buffer data

        """
        data = super().retrieve_all()
        data['returns'] = np.array(self.returns)
        data['advantages'] = np.array(self.advantages)
        return data