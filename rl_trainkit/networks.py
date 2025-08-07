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


class CNN(nn.Module):
    """Convolutional Neural Network feature extractor.

    Automatically adapts to input image size and channels.

    Args:
        input_channels (int): Number of input channels (e.g., 3 for RGB, 1 for grayscale)
        features_dim (int): Output feature dimension
        input_height (int): Height of input image
        input_width (int): Width of input image
    """

    def __init__(self, input_channels, features_dim=512, input_height=84, input_width=84):
        super().__init__()

        # Adaptive architecture based on input size
        if input_height >= 64 and input_width >= 64:
            # Larger images: more conv layers
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
        else:
            # Smaller images: fewer conv layers
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten()
            )

        # Calculate the output dimension after convolutions
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_output_dim = self.conv(sample_input).shape[1]

        # Fully connected layer to get desired feature dimension
        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim, features_dim),
            nn.ReLU()
        )

        self.output_dim = features_dim

    def forward(self, x):
        """Forward pass through CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width)

        Returns:
            torch.Tensor: Feature tensor of shape (batch, features_dim)
        """
        x = self.conv(x)
        x = self.fc(x)
        return x


class ActorCritic(nn.Module):
    """Combined Actor-Critic network with shared feature extractor.

    Args:
        state_dim (int): State dimension (for vector inputs)
        action_dim (int): Action dimension
        hidden_dims (list): Hidden layer dimensions for actor/critic heads
        feature_extractor_type (str): Type of feature extractor ('mlp' or 'cnn')
        shared_dims (list): Hidden dimensions for shared MLP feature extractor
        log_std_min (float): Minimum log standard deviation
        log_std_max (float): Maximum log standard deviation
        cnn_params (dict): Parameters for CNN if using CNN feature extractor
    """

    def __init__(self, state_dim, action_dim, hidden_dims=None,
                 feature_extractor_type='mlp', shared_dims=None,
                 log_std_min=-20, log_std_max=2, cnn_params=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]
        if shared_dims is None:
            shared_dims = [256, 256]

        self.feature_extractor_type = feature_extractor_type
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build feature extractor
        if feature_extractor_type == 'cnn':
            if cnn_params is None:
                raise ValueError("cnn_params must be provided for CNN feature extractor")
            self.feature_extractor = CNN(
                input_channels=cnn_params.get('input_channels', 3),
                features_dim=cnn_params.get('features_dim', 512),
                input_height=cnn_params.get('input_height', 84),
                input_width=cnn_params.get('input_width', 84)
            )
            feature_dim = self.feature_extractor.output_dim
        else:  # mlp
            self.feature_extractor = MLP(state_dim, shared_dims, shared_dims[-1])
            feature_dim = shared_dims[-1]

        # Actor head
        self.actor_head = MLP(feature_dim, hidden_dims, action_dim)

        # Learnable log std for continuous actions
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic_head = MLP(feature_dim, hidden_dims, 1)

    def get_features(self, state):
        """Extract features from state.

        Args:
            state (torch.Tensor): Input state

        Returns:
            torch.Tensor: Extracted features
        """
        return self.feature_extractor(state)

    def get_action_distribution(self, state):
        """Get action distribution from state.

        Args:
            state (torch.Tensor): Input state

        Returns:
            torch.distributions.Normal: Normal distribution for actions
        """
        features = self.get_features(state)
        mean = self.actor_head(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def get_value(self, state):
        """Get value estimate from state.

        Args:
            state (torch.Tensor): Input state

        Returns:
            torch.Tensor: Value estimate
        """
        features = self.get_features(state)
        value = self.critic_head(features)
        return value.squeeze(-1)

    def evaluate_actions(self, state, action):
        """Evaluate actions for given states.

        Args:
            state (torch.Tensor): Input states
            action (torch.Tensor): Actions to evaluate

        Returns:
            tuple: (values, log_probs, entropy)
        """
        features = self.get_features(state)

        # Get value
        values = self.critic_head(features).squeeze(-1)

        # Get action distribution
        mean = self.actor_head(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)

        # Calculate log probs and entropy
        log_probs = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return values, log_probs, entropy

    def forward(self, state, deterministic=False):
        """Forward pass to get action and value.

        Args:
            state (torch.Tensor): Input state
            deterministic (bool): Whether to use deterministic policy

        Returns:
            tuple: (action, value, log_prob)
        """
        features = self.get_features(state)

        # Get value
        value = self.critic_head(features).squeeze(-1)

        # Get action
        mean = self.actor_head(features)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, value, log_prob


class Actor(nn.Module):
    """Actor network for continuous action spaces.

    Args:
        state_dim (int): State dimension
        action_dim (int): Action dimension
        hidden_dims (list): List of hidden layer dimensions
        log_std_min (float): Minimum log standard deviation
        log_std_max (float): Maximum log standard deviation

    """

    def __init__(self, state_dim, action_dim, hidden_dims=None,
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
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

    def __init__(self, state_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.mlp = MLP(state_dim, hidden_dims, 1)

    def forward(self, state):
        """Forward pass.

        Args:
            state (torch.Tensor): State tensor

        Returns:
            value (torch.Tensor): State value

        """
        return self.mlp(state).squeeze(-1)