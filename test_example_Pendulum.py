import gymnasium as gym
import numpy as np
import torch
from rl_trainkit import PPOClip
import time


def test_agent(env, agent, num_episodes=5, render=True):
    """Test trained agent with rendering.

    Args:
        env: Gymnasium environment
        agent: Trained PPO agent
        num_episodes (int): Number of test episodes
        render (bool): Whether to render environment

    """
    episode_returns = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        done = False

        print(f"\nEpisode {episode + 1}")

        while not done:
            # Select action deterministically
            action, _, _ = agent.select_action(state, deterministic=True)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1
            state = next_state

            if render:
                time.sleep(0.01)  # Slow down rendering for visibility

        episode_returns.append(episode_return)
        print(f"Episode length: {episode_length}")
        print(f"Episode return: {episode_return:.2f}")

    print(f"\nAverage return over {num_episodes} episodes: {np.mean(episode_returns):.2f}")
    print(f"Standard deviation: {np.std(episode_returns):.2f}")


def main():
    # Create environment with rendering
    env = gym.make('Pendulum-v1', render_mode='human')

    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create PPO agent
    agent = PPOClip(
        state_dim=state_dim,
        action_dim=action_dim,
        device='cpu'  # Change to 'cuda' if you have GPU
    )

    # Load trained models
    try:
        agent.load_policy_net('models/final_actor.pth')
        agent.load_value_net('models/final_critic.pth')
        print("Models loaded successfully!")
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_Pendulum_example.py first.")
        return

    # Test the agent
    print("\nTesting trained agent...")
    test_agent(env, agent, num_episodes=5, render=True)

    env.close()


if __name__ == "__main__":
    main()