import torch
import torch.optim as optim
import numpy as np
from .networks import Actor, Critic, ActorCritic
from .buffers import TrajectoryBuffer


class PPOClip:
    """PPO agent with clipped objective and optional shared feature extractor.

    Args:
        state_dim (int): State dimension
        action_dim (int): Action dimension
        use_shared_network (bool): Whether to use shared ActorCritic network
        feature_extractor_type (str): 'mlp' or 'cnn' for shared network
        cnn_params (dict): Parameters for CNN if using CNN feature extractor
        clip_range (float): PPO clip range (epsilon in paper)
        value_coef (float): Value loss coefficient (c1 in paper)
        entropy_coef (float): Entropy bonus coefficient (c2 in paper)
        max_kl (float): Maximum KL divergence
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        learning_rate (float): Learning rate for shared optimizer
        actor_lr (float): Actor learning rate (used if not shared)
        critic_lr (float): Critic learning rate (used if not shared)
        update_epochs (int): Number of update epochs
        device (str): Device to use ('cpu' or 'cuda')

    """

    def __init__(self, state_dim, action_dim, use_shared_network=True,
                 feature_extractor_type='mlp', cnn_params=None,
                 clip_range=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_kl=0.01, gamma=0.99, gae_lambda=0.95,
                 learning_rate=3e-4, actor_lr=3e-4, critic_lr=3e-4,
                 update_epochs=10, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_shared_network = use_shared_network
        self.clip_range = clip_range
        self.value_coef = value_coef  # c1 in PPO paper
        self.entropy_coef = entropy_coef  # c2 in PPO paper
        self.max_kl = max_kl
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.device = device

        if use_shared_network:
            # Use combined ActorCritic network with shared feature extractor
            self.actor_critic = ActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                feature_extractor_type=feature_extractor_type,
                cnn_params=cnn_params
            ).to(self.device)

            # Single optimizer for shared network
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

            # For compatibility, create references
            self.actor = None
            self.critic = None
        else:
            # Use separate networks (original implementation)
            self.actor = Actor(state_dim, action_dim).to(self.device)
            self.critic = Critic(state_dim).to(self.device)
            self.actor_critic = None

            # Separate optimizers
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
            if self.use_shared_network:
                action, value, log_prob = self.actor_critic(state_tensor, deterministic)
            else:
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
                if self.use_shared_network:
                    last_value = self.actor_critic.get_value(state_tensor).item()
                else:
                    last_value = self.critic(state_tensor).item()
        else:
            last_value = 0

        # Compute returns and advantages
        self.current_trajectory.compute_returns_and_advantage(last_value)

        # Return completed trajectory and create new one
        trajectory = self.current_trajectory
        self.current_trajectory = TrajectoryBuffer(self.gamma, self.gae_lambda)

        return trajectory

    def update(self, rollout_buffer, logger=None):
        """Update actor and critic networks.

        Args:
            rollout_buffer (RolloutBuffer): Rollout buffer with data
            logger (Logger): Logger for tracking statistics

        """
        for epoch in range(self.update_epochs):
            for batch in rollout_buffer.get_batches():
                self.update_with_batch(batch, logger=logger)

    def update_with_batch(self, batch, logger=None):
        """Update networks with a batch of data.

        Implements the combined loss from PPO paper:
        L^{CLIP+VF+S}(θ) = E_t[L^{CLIP}(θ) - c1 * L^{VF}(θ) + c2 * S[π_θ](s_t)]

        Args:
            batch (dict): Batch of training data
            logger (Logger): Optional logger for statistics
        """
        # Move batch to device
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['log_probs'].to(self.device)
        returns = batch['returns'].to(self.device)
        advantages = batch['advantages'].to(self.device)
        old_values = batch['values'].to(self.device)

        if self.use_shared_network:
            # Combined update for shared network
            # Get current values, log probs, and entropy
            values, log_probs, entropy = self.actor_critic.evaluate_actions(states, actions)

            # Calculate ratio for PPO clip
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped surrogate loss (L^CLIP)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (L^VF) with clipping
            value_pred_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_range, self.clip_range
            )
            value_loss1 = (values - returns).pow(2)
            value_loss2 = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

            # Entropy bonus (S[π_θ](s_t))
            entropy_loss = -entropy.mean()

            # Combined loss: L^{CLIP+VF+S} = L^CLIP - c1*L^VF + c2*S
            # Note: We minimize the loss, so value_loss is positive and entropy_loss is negative
            total_loss = actor_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # Update shared network
            self.optimizer.zero_grad()
            total_loss.backward()
            # Optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

            # Log statistics
            if logger is not None:
                with torch.no_grad():
                    approx_kl = (old_log_probs - log_probs).mean().item()
                    clipped = (ratio != torch.clamp(ratio, 1 - self.clip_range,
                                                    1 + self.clip_range)).float().mean().item()
                    logger.log_update(
                        clipped_ratio=clipped,
                        approx_kl=approx_kl,
                        # actor_loss=actor_loss.item(),
                        # value_loss=value_loss.item(),
                        # entropy=entropy.mean().item(),
                        # total_loss=total_loss.item()
                    )
        else:
            # Separate updates (original implementation)
            # Calculate current log probs and values
            log_probs = self.actor.log_prob(states, actions)
            values = self.critic(states)

            # Calculate ratio and clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)

            # Actor loss
            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # Critic loss (value function clipping)
            value_pred_clipped = old_values + torch.clamp(
                values - old_values, -self.clip_range, self.clip_range
            )
            value_loss1 = (values - returns).pow(2)
            value_loss2 = (value_pred_clipped - returns).pow(2)
            critic_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Log statistics
            if logger is not None:
                with torch.no_grad():
                    approx_kl = (old_log_probs - log_probs).mean().item()
                    clipped = (clipped_ratio != ratio).float().mean().item()
                    logger.log_update(clipped, approx_kl)

    def save_policy_net(self, path):
        """Save policy network.

        Args:
            path (str): Path to save file

        """
        if self.use_shared_network:
            torch.save(self.actor_critic.state_dict(), path)
        else:
            torch.save(self.actor.state_dict(), path)

    def save_value_net(self, path):
        """Save value network.

        Args:
            path (str): Path to save file

        """
        if self.use_shared_network:
            # Save the entire shared network for value net as well
            torch.save(self.actor_critic.state_dict(), path)
        else:
            torch.save(self.critic.state_dict(), path)

    def load_policy_net(self, path):
        """Load policy network.

        Args:
            path (str): Path to load file

        """
        if self.use_shared_network:
            self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.actor.load_state_dict(torch.load(path, map_location=self.device))

    def load_value_net(self, path):
        """Load value network.

        Args:
            path (str): Path to load file

        """
        if self.use_shared_network:
            self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.critic.load_state_dict(torch.load(path, map_location=self.device))