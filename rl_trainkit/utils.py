import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os


class Logger:
    """Logger for training statistics.

    Args:
        verbose (bool): Whether to print logs
        recent_episodes_window (int): Window size for recent episodes

    """

    def __init__(self, verbose=True, recent_episodes_window=100):
        self.verbose = verbose
        self.recent_episodes_window = recent_episodes_window

        # Episode tracking (for moving window)
        self.episode_lengths = deque(maxlen=recent_episodes_window)
        self.episode_returns = deque(maxlen=recent_episodes_window)
        self.episode_terminated = deque(maxlen=recent_episodes_window)

        # Full-history episode tracking (for plotting)
        self.all_episode_returns = []
        self.all_episode_success = []

        # Rollout tracking
        self.rollout_episode_lengths = []
        self.rollout_episode_returns = []
        self.rollout_rewards = []
        self.rollout_clipped_ratios = []
        self.rollout_kls = []

        # History of rollout‐level statistics (for plotting)
        self.rollout_indices = []
        self.rollout_means = []
        self.rollout_cis95 = []

        # Global counters
        self.total_episodes = 0
        self.total_timesteps = 0
        self.total_rollouts = 0

        self.start_time = time.time()
        self.last_log_time = time.time()
        self.separator_line_length = 35

    def log_episode(self, length, return_val, terminated):
        """Log episode statistics.

        Args:
            length (int): Episode length
            return_val (float): Episode return
            terminated (bool): Whether episode terminated

        """
        self.episode_lengths.append(length)
        self.episode_returns.append(return_val)
        self.episode_terminated.append(terminated)

        # full history
        self.all_episode_returns.append(return_val)
        self.all_episode_success.append(int(terminated))

        self.total_episodes += 1
        self.total_timesteps += length

        # Also add to roll out tracking
        self.rollout_episode_lengths.append(length)
        self.rollout_episode_returns.append(return_val)

    def log_step(self, reward):
        """Log step statistics.

        Args:
            reward (float): Step reward

        """
        self.rollout_rewards.append(reward)

    def log_update(self, clipped_ratio, kl):
        """Log update statistics.

        Args:
            clipped_ratio (float): Clipped ratio
            kl (float): KL divergence

        """
        self.rollout_clipped_ratios.append(clipped_ratio)
        self.rollout_kls.append(kl)

    def print_log(self, total_timesteps_target, clip_range):
        """Print training log.

        Args:
            total_timesteps_target (int): Target total timesteps
            clip_range (float): PPO clip range

        """
        if not self.verbose:
            # still snapshot rollout if needed
            self._snapshot_rollout_stats()
            self._clear_rollout_statistics()
            return

        current_time = time.time()
        time_elapsed = current_time - self.last_log_time
        self.last_log_time = current_time

        # Rollout statistics
        mean_ep_len = np.mean(self.rollout_episode_lengths) if self.rollout_episode_lengths else 0
        mean_ep_return = np.mean(self.rollout_episode_returns) if self.rollout_episode_returns else 0
        mean_reward = np.mean(self.rollout_rewards) if self.rollout_rewards else 0
        mean_clipped_ratio = np.mean(self.rollout_clipped_ratios) if self.rollout_clipped_ratios else 0
        approx_kl = np.mean(self.rollout_kls) if self.rollout_kls else 0
        rollout_len = len(self.rollout_rewards)

        # Recent episodes statistics
        recent_mean_ep_len = np.mean(self.episode_lengths) if self.episode_lengths else 0
        recent_mean_ep_return = np.mean(self.episode_returns) if self.episode_returns else 0
        success_rate = np.mean(self.episode_terminated) if self.episode_terminated else 0

        print("\n" + "=" * self.separator_line_length)
        print("ROLLOUT STATISTICS")
        print("-" * self.separator_line_length)
        print(f"  mean_ep_len:        {mean_ep_len:.2f}")
        print(f"  mean_ep_return:     {mean_ep_return:.2f}")
        print(f"  mean_reward:        {mean_reward:.4f}")
        print(f"  mean_clipped_ratio: {mean_clipped_ratio:.4f}")
        print(f"  approx_kl:          {approx_kl:.6f}")
        print(f"  rollout_len:        {rollout_len}")
        print(f"  time_elapsed:       {time_elapsed:.2f}s")

        print("\n" + f"RECENT {self.recent_episodes_window} EPISODES")
        print("-" * self.separator_line_length)
        print(f"  mean_ep_len:        {recent_mean_ep_len:.2f}")
        print(f"  mean_ep_return:     {recent_mean_ep_return:.2f}")
        print(f"  success_rate:       {success_rate:.2%}")

        print("\n" + "TRAINER STATISTICS")
        print("-" * self.separator_line_length)
        print(f"  episodes_count:       {self.total_episodes}")
        print(f"  timesteps_count:      {self.total_timesteps}")
        print(f"  rollouts_count:       {self.total_rollouts}")
        print(f"  total_timesteps:    {total_timesteps_target}")
        print(f"  clip_range:         {clip_range}")
        print("=" * self.separator_line_length + "\n")

        # snapshot & clear
        self._snapshot_rollout_stats()
        self._clear_rollout_statistics()

    def _snapshot_rollout_stats(self):
        """Record mean & 95% CI of the just‐finished rollout."""
        if not self.rollout_episode_returns:
            return
        m = np.mean(self.rollout_episode_returns)
        std = float(np.std(self.rollout_episode_returns, ddof=1))
        n = len(self.rollout_episode_returns)
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0

        self.rollout_indices.append(self.total_rollouts)
        self.rollout_means.append(m)
        self.rollout_cis95.append(ci95)

    def _clear_rollout_statistics(self):
        self.rollout_episode_lengths.clear()
        self.rollout_episode_returns.clear()
        self.rollout_rewards.clear()
        self.rollout_clipped_ratios.clear()
        self.rollout_kls.clear()

class Visualizer:
    """Visualization of training curves from a Logger."""

    def __init__(self, logger: Logger, output_dir: str = "./output"):
        self.logger = logger
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Set style for better-looking plots
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def plot_ckpt(self, timestep: int):
        """Generate & save all three plots at a checkpoint."""
        self._plot_episode_returns(timestep)
        self._plot_success_rates(timestep)
        self._plot_rollout_returns(timestep)

    def _plot_episode_returns(self, timestep: int):
        x = list(range(1, len(self.logger.all_episode_returns) + 1))
        y = self.logger.all_episode_returns
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Episodes")
        plt.ylabel("Return")
        plt.title(f"Episode Returns @ {timestep} ts")
        path = os.path.join(self.output_dir, f"episode_returns_{timestep}.png")
        plt.savefig(path)
        plt.close()

    def _plot_success_rates(self, timestep: int):
        window = self.logger.recent_episodes_window
        flags = self.logger.all_episode_success
        rates = []
        for i in range(len(flags)):
            start = max(0, i - window + 1)
            rates.append(np.mean(flags[start : i + 1]))
        x = list(range(1, len(rates) + 1))
        plt.figure()
        plt.plot(x, rates)
        plt.xlabel("Episodes")
        plt.ylabel("Success Rate")
        plt.title(f"Success Rate (window={window}) @ {timestep} ts")
        path = os.path.join(self.output_dir, f"success_rates_{timestep}.png")
        plt.savefig(path)
        plt.close()

    def _plot_rollout_returns(self, timestep: int):
        x = self.logger.rollout_indices
        y = self.logger.rollout_means
        ci = self.logger.rollout_cis95
        if not x:
            return
        plt.figure()
        plt.errorbar(x, y, yerr=ci, fmt='o-')
        plt.xlabel("Rollouts")
        plt.ylabel("Mean Return")
        plt.title(f"Rollout Mean Returns ±95% CI @ {timestep} ts")
        path = os.path.join(self.output_dir, f"rollout_returns_{timestep}.png")
        plt.savefig(path)
        plt.close()