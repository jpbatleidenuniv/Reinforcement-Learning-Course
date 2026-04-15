from collections.abc import Sequence
from torch import Tensor
from numpy.typing import NDArray
from config import AgentConfig, NNConfig
import torch.nn as nn
import torch


class PINetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        features: Sequence[int],
    ):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, features[0]), nn.ReLU())
        self.body = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(features[i], features[i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(features) - 1)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(features[-1], output_dim), nn.Softmax(dim=-1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        for layer in self.body:
            x = layer(x)
        return self.output_layer(x)


class Agent:
    def __init__(self, agent_cfg: AgentConfig, nn_cfg: NNConfig) -> None:
        self.agent_cfg = agent_cfg
        self.nn_cfg = nn_cfg
        self.policy = PINetwork(
            nn_cfg.input_dim,
            nn_cfg.output_dim,
            nn_cfg.features,
        )
        self.optimizer = self._build_optimizer(nn_cfg.optim)
        # Define arrays of rollout monte carlo
        self.log_probs: list[Tensor] = []
        self.rewards: list[float] = []

    def select_action(self, obs: NDArray) -> tuple[int, Tensor]:
        # First we make predictions fro policy at state 'obs'
        pi_s: Tensor = self.policy(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        ).squeeze(0)  # Softmaxed dist of 2 possible actions
        dist = torch.distributions.Categorical(pi_s)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob

    def _build_optimizer(self, optimizer: str):
        registry = {
            "AdamW": torch.optim.AdamW(self.policy.parameters(), lr=self.nn_cfg.lr),
            "Adam": torch.optim.Adam(self.policy.parameters(), lr=self.nn_cfg.lr),
            "RMSprop": torch.optim.RMSprop(self.policy.parameters(), lr=self.nn_cfg.lr),
        }
        if optimizer not in registry:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        return registry[optimizer]

    @property
    def _returns(self):
        """Tensor of G_1 ... G_T"""
        T = len(self.rewards)
        rewards = torch.tensor(self.rewards)
        gammas = torch.pow(
            torch.tensor(self.agent_cfg.gamma, dtype=torch.float32),
            torch.arange(T, dtype=torch.float32),
        )
        discounted = gammas * rewards

        # Discounted cumulative returns from t=0 to t=T (episode length)
        returns: Tensor = torch.cumsum(discounted.flip(0), dim=0).flip(0) / gammas
        return returns

    def update(self):
        self.optimizer.zero_grad()
        T = len(self.rewards)
        objectives = self._returns  # T targets, this is G_t
        objectives = (objectives - objectives.mean()) / (objectives.std() + 1e-8)
        assert T == len(objectives), f"T = {T}, len(G_T) = {len(objectives)}"
        log_probs = torch.stack(self.log_probs)
        loss = -1 * (log_probs * objectives).sum()

        loss.backward()
        self.optimizer.step()

        info = {"loss": loss.item(), "step": T, "episode_return": sum(self.rewards)}

        # We wipe the episode information for the next episode
        self.log_probs.clear()
        self.rewards.clear()

        return info
