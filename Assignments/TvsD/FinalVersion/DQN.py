import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym


class QNetwork(nn.Module):
    """
    Fully-connected Q-network that maps observations to Q-values for each action.

    Architecture: Linear -> ReLU (repeated for each hidden layer) -> Linear output.
    Defaults to two hidden layers of width 64 if none are specified.
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

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input shape: (batch, obs_dim). Output shape: (batch, action_dim)."""
        return self.net(x)


class DQNAgent:
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
        exploration_strategy: str = "epsilon_greedy",
        epsilon: float = 0.05,
        temperature: float = 1.0,
        gamma: float = 0.99,
        device: str = "cpu",
        use_target_network: bool = False,
        target_update_every: int = 100,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)

        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.temperature = temperature

        self.use_target_network = use_target_network
        self.target_update_every = target_update_every
        self.num_updates = 0  # Counts train_step calls; used to schedule target network syncs

        # Online Q-network: updated every train_step
        self.q_network = QNetwork(obs_dim=obs_dim, action_dim=action_dim,
                                  hidden_layers=hidden_layers).to(self.device)

        if self.use_target_network:
            # Target network: identical architecture, weights periodically copied from q_network
            self.target_network = QNetwork(obs_dim=obs_dim, action_dim=action_dim,
                                           hidden_layers=hidden_layers).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval() 
        else:
            self.target_network = None

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.MSELoss()

    def select_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Select actions for a batch of observations using the configured exploration strategy.

        Args:
            obs: Shape (num_envs, obs_dim)
        Returns:
            actions: Shape (num_envs,) as int64
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_values = self.q_network(obs_tensor)  # (num_envs, action_dim)

        if self.exploration_strategy == "epsilon_greedy":
            return self._select_actions_epsilon(q_values)
        elif self.exploration_strategy == "softmax":
            return self._select_actions_softmax(q_values)

        raise ValueError(f"Exploration strategy typo: {self.exploration_strategy}")

    def _select_actions_epsilon(self, q_values: torch.Tensor) -> np.ndarray:
        """
        Epsilon-greedy action selection: with probability epsilon choose a random action,
        otherwise choose the greedy (argmax) action.
        """
        batch_size = q_values.shape[0]

        greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()
        random_actions = np.random.randint(0, self.action_dim, size=batch_size)

        # For each env, independently decide whether to explore
        explore_mask = np.random.rand(batch_size) < self.epsilon
        actions = np.where(explore_mask, random_actions, greedy_actions)

        return actions.astype(np.int64)

    def _select_actions_softmax(self, q_values: torch.Tensor) -> np.ndarray:
        """
        Softmax action selection: sample actions proportionally to
        exp(Q / temperature). Higher temperature -> more uniform exploration.
        """
        scaled_q = q_values / self.temperature
        action_probs = torch.softmax(scaled_q, dim=1)

        action_probs_np = action_probs.cpu().numpy()
        actions = np.array([
            np.random.choice(self.action_dim, p=action_probs_np[i])
            for i in range(action_probs_np.shape[0])
        ])

        return actions.astype(np.int64)

    def train_step(
            self,
            obs: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_obs: np.ndarray,
            dones: np.ndarray,
    ) -> float:
        """
        Perform one gradient update on the Q-network using a batch of transitions.
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s, a) for the actions that were actually taken
        q_values = self.q_network(obs_tensor)                                    # (batch, action_dim)
        chosen_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  # (batch,)

        with torch.no_grad():
            # Use target network if available, otherwise bootstrap from the online network
            if self.use_target_network:
                next_q_values = self.target_network(next_obs_tensor)
            else:
                next_q_values = self.q_network(next_obs_tensor)

            max_next_q_values = next_q_values.max(dim=1)[0]
            # Mask out terminal states so no future reward is added after episode end
            targets = rewards_tensor + self.gamma * (1.0 - dones_tensor) * max_next_q_values

        loss = self.loss_fn(chosen_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.num_updates += 1

        # Periodically copy online network weights to the target network
        if self.use_target_network and self.num_updates % self.target_update_every == 0:
            self.update_target_network()

        return float(loss.item())

    def update_target_network(self):
        """copy all weights from the online Q-network to the target network."""
        if self.target_network is not None:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def select_greedy_actions(self, obs: np.ndarray) -> np.ndarray:
        """Greedy action selection. Used during evaluation."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            q_values = self.q_network(obs_tensor)

        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        return actions.astype(np.int64)

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
