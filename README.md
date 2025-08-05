# rl_trainkit

Implementations of Reinforcement learning algorithms in PyTorch.

- Support only **PPO-clip** for now
- Support only **vector observation**, **continues action** for now

## Quick start

### Requirements

> gymnasium torch numpy tqdm

### Import as package

Place the `rl_trainkit` folder from the repository in the same directory as the training script.
In the training script, add the following import statement:

```python
from rl_trainkit import PPOClip, OnPolicyTrainer
```

Detailed usage and project structure explanation, see [Doc](documentation/Doc.md)

Training script example: refer to [train_example_Pendulum.py](train_example_Pendulum.py)

## TODO

- [ ] Add support for a shared feature extraction network for actor-critic.
- [ ] Add thread management to improve the efficiency of sampling, data processing, and network updates; introduce support for callback functions.
- [ ] Add data visualization scripts (similar to TensorBoard) for monitoring training metrics, losses, rewards, etc.
- [ ] Expand documentation: Training workflow
- [ ] Add support for other RL algorithms.

## References

- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Openai, “Proximal Policy Optimization Algorithms,” Aug. 2017. Available: [https://arxiv.org/pdf/1707.06347](https://arxiv.org/pdf/1707.06347)
- OpenAI [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
- DLR-RM [Stable Baseline3](https://stable-baselines3.readthedocs.io/en/master/index.html)
- Eric Yang Yu [Coding PPO from Scratch with PyTorch](https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8)
- S. Huang et al., "The 37 Implementation Details of Proximal Policy Optimization," ICLR Blog Track, Mar. 25, 2022. [Online]. Available: [https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
