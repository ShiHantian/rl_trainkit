import gymnasium as gym
import torch
from rl_trainkit import PPOClip, OnPolicyTrainer
import os


def main():
    # Create environment
    env = gym.make('Pendulum-v1')

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create PPO agent
    agent = PPOClip(
        state_dim=state_dim,
        action_dim=action_dim,
        clip_range=0.2,
        value_clip_range=0.1,
        gamma=0.99,
        gae_lambda=0.95,
        actor_lr=3e-4,
        critic_lr=3e-4,
        update_epochs=10,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create trainer
    trainer = OnPolicyTrainer(
        environment=env,
        agent=agent,
        total_timesteps=300_000,
        threshold_rollout_length=2048,
        max_episode_len=200,
        batch_size=64,
        num_checkpoints=4,
        mini_ckpt_period=4096,
        verbose=True
    )

    # Train the agent
    trainer.train()

    env.close()


if __name__ == "__main__":
    main()