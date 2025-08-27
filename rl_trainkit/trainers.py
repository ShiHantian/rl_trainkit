from tqdm import tqdm
import numpy as np
import os
from .buffers import RolloutBuffer
from .utils import Logger, Visualizer


class OnPolicyTrainer:
    """On-policy trainer for PPO algorithm.

    Args:
        environment: Gymnasium environment
        agent: PPO agent
        total_timesteps (int): Total training timesteps
        threshold_rollout_length (int): Minimum rollout length
        max_episode_len (int): Maximum episode length
        batch_size (int): Batch size for training
        batch_num (int): Number of batches (if None, use batch_size)
        verbose (bool): Whether to print logs
        num_checkpoints (int): How many intermediate checkpoints to save (saved at fractions i/(num_checkpoints+1))
    """

    def __init__(
        self,
        environment,
        agent,
        total_timesteps=1_000_000,
        threshold_rollout_length=2048,
        max_episode_len=1024,
        batch_size=64,
        num_checkpoints=3,
        batch_num=None,
        verbose=True,
        mini_ckpt_period=None,
    ):
        self.env = environment
        self.agent = agent
        self.total_timesteps = total_timesteps
        self.threshold_rollout_length = threshold_rollout_length
        self.max_episode_len = max_episode_len
        self.batch_size = batch_size
        self.batch_num = batch_num

        # Logger and Visualizer
        self.logger = Logger(verbose=verbose)
        self.visualizer = Visualizer(self.logger)
        # Mini‐checkpoint parameters
        if mini_ckpt_period is not None:
            self.mini_ckpt_period = mini_ckpt_period
        else:
            self.mini_ckpt_period = total_timesteps/1000
        self._last_mini_ckpt = 0

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


    def collect_one_episode(self) -> tuple[RolloutBuffer, dict]:
        """Collect one complete episode.
        
        Returns:
            trajectory_buffer(TrajectoryBuffer): Complete trajectory for the episode
            episode_info(dict): Dictionary containing episode statistics
        """
        # Initialize episode
        state, _ = self.env.reset()
        episode_length = 0
        episode_return = 0
        
        # Collect episode data
        while episode_length < self.max_episode_len:
            # Select action
            action, value, log_prob = self.agent.select_action(state)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.agent.current_trajectory.push_one_step(
                state, action, reward, next_state, done, value, log_prob
            )
            
            # Update episode tracking
            episode_length += 1
            episode_return += reward
            
            # Check if episode ended
            if done:
                break
            else:
                state = next_state
        
        # Finish trajectory
        if done:
            trajectory_buffer = self.agent.finish_trajectory(None)
        else:
            trajectory_buffer = self.agent.finish_trajectory(next_state)
        
        # Clear the trajectory buffer
        self.agent.current_trajectory.clear()
        
        # Prepare episode info
        episode_info = {
            'length': episode_length,
            'return': episode_return,
            'terminated': terminated if done else False
        }
        
        return trajectory_buffer, episode_info

    def collect_one_rollout(self) -> RolloutBuffer:
        """Collect one rollout, episode by episode.

        Returns:
            rollout_buffer (RolloutBuffer): one rollout of collected data
        """
        # Create rollout buffer
        rollout_buffer = RolloutBuffer(
            self.threshold_rollout_length,
            self.batch_size,
            self.batch_num
        )
        
        rollout_length = 0
        
        # Collect episodes until rollout buffer is full
        while len(rollout_buffer) < self.threshold_rollout_length:
            # Collect one episode
            trajectory_buffer, episode_info = self.collect_one_episode()
            
            # Add trajectory to rollout buffer
            rollout_buffer.concat(trajectory_buffer=trajectory_buffer)
            
            # Log episode
            self.logger.log_episode(
                length=episode_info['length'], 
                return_val=episode_info['return'], 
                terminated=episode_info['terminated']
            )
            
            # Update rollout length
            rollout_length += episode_info['length']

            # Update progress bar
            self.pbar.update(episode_info['length'])

        # Log rollout
        self.logger.log_rollout(rollout_length)
        return rollout_buffer

    def train(self):
        """Train the agent.

        """

        while self.logger.total_timesteps < self.total_timesteps:
            # Collect one rollout
            rollout_buffer = self.collect_one_rollout()

            # Update networks
            self.agent.update(rollout_buffer, self.logger)

            # Print log
            self.logger.total_rollouts += 1
            self.logger.print_log(self.total_timesteps, self.agent.clip_range)

            # === mini-checkpoint and visualizing ===
            # mini‐checkpoint plots
            if self.logger.total_timesteps >= (self._last_mini_ckpt + self.mini_ckpt_period):
                self.visualizer.plot_ckpt(self.logger.total_timesteps)
                self._last_mini_ckpt += self.mini_ckpt_period

            # === checkpoint and best model saving ===
            # 1. Checkpoint saves at checkpoint milestones
            for idx, milestone in enumerate(self.checkpoint_timesteps):
                if (not self.checkpoints_saved[idx]) and (self.logger.total_timesteps >= milestone):
                    ckpt_path = os.path.join("models", f"pi_ckpt{idx+1}.pth")
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

        # === Total timesteps reached, training completed ===
        # Save the trained models
        os.makedirs("models", exist_ok=True)
        final_actor_path = os.path.join("models", "final_actor.pth")
        final_critic_path = os.path.join("models", "final_critic.pth")
        self.agent.save_policy_net(final_actor_path)
        self.agent.save_value_net(final_critic_path)
        print("Final models saved!")

        # final checkpoint plots
        self.visualizer.plot_ckpt(self.logger.total_timesteps)

        # Close progress bar
        self.pbar.close()
        print("\nTraining completed!")