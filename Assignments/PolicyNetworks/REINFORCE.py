import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym


class PolicyNetwork(nn.Module):
    """
    Fully-connected Q-network that maps observations to probabilities for each action.
    """

    def __init__(self, obs_dim: int = 4, action_dim: int = 2, hidden_layers: list[int] | None = None):
        super().__init__()

        layers = []
        in_dim = obs_dim

        if hidden_layers is None:
            hidden_layers = [64, 64]

        # Build hidden layers dynamically from the provided width list
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Final linear layer outputs one Q-value per action (no activation)
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Softmax(action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input shape: (batch, obs_dim). Output shape: (batch, action_dim)."""
        return self.net(x)


class REINFORCEAgent:
    """
    DQN agent that wraps a QNetwork with action selection, training, and evaluation.

    Supports:
      - Epsilon-greedy and softmax exploration strategies
      - Optional target network (frozen copy of Q-network, periodically synced)
      - Vectorized action selection for multiple parallel environments
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layers: list[int],
        learning_rate: float,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)

        # Online Q-network: updated every train_step
        self.q_network = QNetwork(obs_dim=obs_dim, action_dim=action_dim,
                                  hidden_layers=hidden_layers).to(self.device)

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.MSELoss()

    def select_actions(self, obs: np.ndarray, greedy=False) -> tuple[int, torch.Tensor]:
        """
        Select actions for a batch of observations using the configured exploration strategy.

        Args:
            obs: Shape (num_envs, obs_dim)
        Returns:
            actions: Shape (num_envs,) as int64
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        actions_p = self.q_network(obs_tensor)
        if not greedy:
            action = int(np.random.choice([0,1], actions_p.detach().numpy()))

        else:
            action = int(np.argmax(actions_p.detach().numpy()))
        return action, actions_p[action]

    def train_step(self,
                   rewards: np.ndarray,
                   action_probs: torch.Tensor) -> None:
        
        likelihoods = torch.log(action_probs)
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        gamma_matrix = torch.triu(input=torch.tensor(self.gamma), diagonal=0)
        G_t = torch.dot(gamma_matrix, r_tensor)
        loss = torch.sum(G_t * likelihoods)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def evaluate(self, n_episodes: int = 10, seed: int = 123) -> float:
        """
        Evaluate the current policy greedily over n_episodes independent episodes.

        Returns the mean undiscounted episode return.
        """
        env = gym.make("CartPole-v1")
        returns = []

        for episode in range(n_episodes):
            obs, _ = env.reset(seed=seed + episode)
            obs = np.expand_dims(obs, axis=0)  # Add batch dimension: (1, obs_dim)

            done = False
            episode_return = 0.0

            while not done:
                action = self.select_greedy_actions(obs)[0]
                next_obs, reward, terminated, truncated, _ = env.step(int(action))

                done = terminated or truncated
                episode_return += reward

                obs = np.expand_dims(next_obs, axis=0)

            returns.append(episode_return)

        env.close()
        return float(np.mean(returns))
