from tqdm import tqdm
import numpy as np
import os
from .buffers import RolloutBuffer
from .utils import Logger


class OnPolicyTrainer:
    """On-policy trainer for PPO algorithm.

    Args:
        environment: Gymnasium environment
        agent: PPO agent
        total_timesteps (int): Total training timesteps
        threshold_rollout_length (int): Minimum rollout length
        max_episode_len (int): Maximum episode length
        device (str): Device to use ('cpu' or 'cuda')
        batch_size (int): Batch size for training
        batch_num (int): Number of batches (if None, use batch_size)
        verbose (bool): Whether to print logs
        num_checkpoints (int): How many intermediate checkpoints to save (saved at fractions i/(num_checkpoints+1))
    """

    def __init__(self, environment, agent, total_timesteps=1_000_000,
                 threshold_rollout_length=2048, max_episode_len=1000,
                 batch_size=64,
                 num_checkpoints=3,
                 batch_num=None, verbose=True):
        self.env = environment
        self.agent = agent
        self.total_timesteps = total_timesteps
        self.threshold_rollout_length = threshold_rollout_length
        self.max_episode_len = max_episode_len
        self.batch_size = batch_size
        self.batch_num = batch_num

        # Logger
        self.logger = Logger(verbose=verbose)

        # Progress bar
        self.pbar = tqdm(total=total_timesteps, desc="Training Progress")

        # Checkpoint logic
        os.makedirs("models", exist_ok=True)
        self.num_checkpoints = num_checkpoints
        # Checkpoint timesteps are at i/(num_checkpoints+1) ratio of total timesteps
        self.checkpoint_timesteps = [
            total_timesteps * i / (self.num_checkpoints + 1)
            for i in range(1, self.num_checkpoints + 1)
        ]
        self.checkpoints_saved = [False] * self.num_checkpoints  # flags for ckpt_q1...qN

        # Best return tracking
        self.best_mean_return = -float("inf")


    def train(self):
        """Train the agent.

        """
        # Initialize
        state, _ = self.env.reset()
        episode_length = 0
        episode_return = 0

        while self.logger.total_timesteps < self.total_timesteps:
            # Create rollout buffer
            rollout_buffer = RolloutBuffer(
                self.threshold_rollout_length,
                self.batch_size,
                self.batch_num
            )

            # Collect rollout
            while len(rollout_buffer) < self.threshold_rollout_length:
                # Select action
                action, value, log_prob = self.agent.select_action(state)

                # Step environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store transition
                self.agent.current_trajectory.push_one_step(
                    state, action, reward, next_state, done, value, log_prob
                )

                # Log step
                self.logger.log_step(reward)

                # Update episode tracking
                episode_length += 1
                episode_return += reward

                # Check if episode ended
                if done or episode_length >= self.max_episode_len:
                    # Finish trajectory
                    if done:
                        trajectory = self.agent.finish_trajectory(None)
                    else:
                        trajectory = self.agent.finish_trajectory(next_state)

                    # Add to roll out buffer
                    rollout_buffer.concat(trajectory)

                    # Log episode
                    self.logger.log_episode(episode_length, episode_return, terminated)

                    # Reset episode
                    state, _ = self.env.reset()
                    episode_length = 0
                    episode_return = 0
                else:
                    state = next_state

                # Update progress bar
                self.pbar.update(1)

                # Check if we've reached total timesteps
                if self.logger.total_timesteps >= self.total_timesteps:
                    break

            # If current trajectory has data, finish it
            if len(self.agent.current_trajectory) > 0:
                trajectory = self.agent.finish_trajectory(state)
                rollout_buffer.concat(trajectory)

            # Update networks
            self.agent.update_with_rollout(rollout_buffer, self.logger)

            # Print log
            self.logger.total_rollouts += 1
            self.logger.print_log(self.total_timesteps, self.agent.clip_range)

            # === checkpoint and best model saving ===
            # 1. Checkpoint saves at checkpoint milestones
            for idx, milestone in enumerate(self.checkpoint_timesteps):
                if (not self.checkpoints_saved[idx]) and (self.logger.total_timesteps >= milestone):
                    ckpt_name = f"pi_ckpt{idx + 1}.pth"
                    ckpt_path = os.path.join("models", ckpt_name)
                    self.agent.save_policy_net(ckpt_path)
                    self.checkpoints_saved[idx] = True

            # 2. Best recent-return model
            if len(self.logger.episode_returns) > 0:
                recent_mean_return = np.mean(self.logger.episode_returns)
                if recent_mean_return > self.best_mean_return:
                    self.best_mean_return = recent_mean_return
                    best_path = os.path.join("models", "best_return_pi_ckpt.pth")
                    self.agent.save_policy_net(best_path)

            # Clear buffers
            rollout_buffer.clear()

        # Save the trained models
        os.makedirs("models", exist_ok=True)
        final_actor_path = os.path.join("models", "final_actor.pth")
        final_critic_path = os.path.join("models", "final_critic.pth")
        self.agent.save_policy_net(final_actor_path)
        self.agent.save_value_net(final_critic_path)
        print("Final models saved!")

        # Close progress bar
        self.pbar.close()
        print("\nTraining completed!")