import torch
import torch.nn as nn

class MLP(nn.Module):
    """Multi-Layer Perceptron network.

    Args:
        input_dim (int): Input dimension
        hidden_dims (list): List of hidden layer dimensions
        output_dim (int): Output dimension
        activation (nn.Module): Activation function

    """

    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Actor(nn.Module):
    """Actor network for continuous action spaces.

    Args:
        state_dim (int): State dimension
        action_dim (int): Action dimension
        hidden_dims (list): List of hidden layer dimensions
        log_std_min (float): Minimum log standard deviation
        log_std_max (float): Maximum log standard deviation

    """

    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64],
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_dims, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        """Forward pass.

        Args:
            state (torch.Tensor): State tensor

        Returns:
            mean (torch.Tensor): Action mean
            std (torch.Tensor): Action standard deviation

        """
        mean = self.mlp(state)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def get_distribution(self, state):
        """Get action distribution.

        Args:
            state (torch.Tensor): State tensor

        Returns:
            dist (torch.distributions.Normal): Normal distribution

        """
        mean, std = self.forward(state)
        return torch.distributions.Normal(mean, std)

    def log_prob(self, state, action):
        """Calculate log probability of action.

        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor

        Returns:
            log_prob (torch.Tensor): Log probability

        """
        dist = self.get_distribution(state)
        return dist.log_prob(action).sum(dim=-1)


class Critic(nn.Module):
    """Critic network for value function estimation.

    Args:
        state_dim (int): State dimension
        hidden_dims (list): List of hidden layer dimensions

    """

    def __init__(self, state_dim, hidden_dims=[64, 64]):
        super().__init__()
        self.mlp = MLP(state_dim, hidden_dims, 1)

    def forward(self, state):
        """Forward pass.

        Args:
            state (torch.Tensor): State tensor

        Returns:
            value (torch.Tensor): State value

        """
        return self.mlp(state).squeeze(-1)