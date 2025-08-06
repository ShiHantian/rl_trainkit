import torch
import torch.optim as optim
import numpy as np
from .networks import Actor, Critic
from .buffers import TrajectoryBuffer


class PPOClip:
    """PPO agent with clipped objective.

    Args:
        state_dim (int): State dimension
        action_dim (int): Action dimension
        clip_range (float): PPO clip range
        value_clip_range (float): Value clip range
        target_kl (float): Maximum KL divergence
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        actor_lr (float): Actor learning rate
        critic_lr (float): Critic learning rate
        update_epochs (int): Number of update epochs
        device (str): Device to use ('cpu' or 'cuda')

    """

    def __init__(self, state_dim, action_dim, clip_range=0.2, value_clip_range=0.2, target_kl=0.01,
                 gamma=0.99, gae_lambda=0.95, actor_lr=3e-4, critic_lr=3e-4,
                 update_epochs=10, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.target_kl = target_kl
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.device = device

        # Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Current trajectory buffer
        self.current_trajectory = TrajectoryBuffer(gamma, gae_lambda)

    def select_action(self, state, deterministic=False):
        """Select action given state.

        Args:
            state (np.ndarray): Current state
            deterministic (bool): Whether to use deterministic policy

        Returns:
            action (np.ndarray): Selected action
            value (float): State value
            log_prob (float): Action log probability

        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get action distribution
            dist = self.actor.get_distribution(state_tensor)

            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state_tensor)

        return action.squeeze(0).cpu().numpy(), value.item(), log_prob.item()

    def finish_trajectory(self, last_state):
        """Finish current trajectory and compute returns.

        Args:
            last_state (np.ndarray): Last state of trajectory

        Returns:
            trajectory (TrajectoryBuffer): Completed trajectory buffer

        """
        # Compute last value if trajectory not done
        if last_state is not None:
            state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                last_value = self.critic(state_tensor).item()
        else:
            last_value = 0

        # Compute returns and advantages
        self.current_trajectory.compute_returns_and_advantage(last_value)

        # Return completed trajectory and create new one
        trajectory = self.current_trajectory
        self.current_trajectory = TrajectoryBuffer(self.gamma, self.gae_lambda)

        return trajectory

    def update_with_rollout(self, rollout_buffer, logger=None):
        """Update actor and critic networks with the rollout buffer.

        Args:
            rollout_buffer (RolloutBuffer): Rollout buffer with data
            logger (Logger): Logger for tracking statistics

        """
        for epoch in range(self.update_epochs):
            for batch in rollout_buffer.get_batches():
                early_stop = self.update_with_batch(batch, logger)
                if early_stop:
                    return

    def update_with_batch(self, batch, logger=None):
        """Update actor and critic networks with the rollout buffer.

        Args:
            batch (dict): One batch of data from the shuffled rollout buffer
            logger (Logger): Logger for tracking statistics

        """
        # Move the batch to device
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        returns = batch['returns'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        old_values = batch['values'].to(self.device)

        # Calculate current log probs and values
        log_probs = self.actor.log_prob(states, actions)
        values = self.critic(states)

        # Calculate ratio and clipped objective
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

        # Actor loss
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Critic loss (clipped value objective)
        value_pred_clipped = old_values + \
            torch.clamp(values - old_values, -self.value_clip_range, self.value_clip_range)
        value_loss_unclipped = (values - returns).pow(2)
        value_loss_clipped = (value_pred_clipped - returns).pow(2)
        critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Log statistics
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean().item()
            clipped = (clipped_ratio != ratio).float().mean().item()
            logger.log_update(clipped, approx_kl)

        # Check for early stop for too big KL divergence
        early_stop = False
        if approx_kl > 1.5 * self.target_kl:
            early_stop = True

        return early_stop

    def save_policy_net(self, path):
        """Save policy network.

        Args:
            path (str): Path to save file

        """
        torch.save(self.actor.state_dict(), path)

    def save_value_net(self, path):
        """Save value network.

        Args:
            path (str): Path to save file

        """
        torch.save(self.critic.state_dict(), path)

    def load_policy_net(self, path):
        """Load policy network.

        Args:
            path (str): Path to load file

        """
        self.actor.load_state_dict(torch.load(path, map_location=self.device))

    def load_value_net(self, path):
        """Load value network.

        Args:
            path (str): Path to load file

        """
        self.critic.load_state_dict(torch.load(path, map_location=self.device))