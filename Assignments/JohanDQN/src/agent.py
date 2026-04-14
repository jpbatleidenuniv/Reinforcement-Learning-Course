from collections.abc import Sequence
from torch import Tensor
from numpy.typing import NDArray
from config import AgentConfig, NNConfig
import numpy as np
import torch.nn as nn
import torch
import random


def argmax(x) -> Tensor:
    """Argmax with random tie breaking, tensor style"""
    arg_maxes = torch.where(x == torch.max(x))[0]
    random_index = torch.randint(0, len(arg_maxes), (1,))
    return arg_maxes[random_index]


def softmax(x: Tensor, temp: float):
    """Computes the softmax of vector x with temperature parameter 'temp'"""
    x = x / temp  # scale by temperature
    z = x - torch.max(x)
    return torch.exp(z) / torch.sum(
        torch.exp(z)
    )  # compute softmax


class DQN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        features: Sequence[int],
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, features[0]), nn.ReLU()
        )
        self.body = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(features[i], features[i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(features) - 1)
            ]
        )
        self.output_layer = nn.Linear(
            features[-1], output_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        for layer in self.body:
            x = layer(x)
        return self.output_layer(x)


class Agent:
    def __init__(
        self, agent_cfg: AgentConfig, nn_cfg: NNConfig
    ) -> None:
        self.agent_cfg = agent_cfg
        self.nn_cfg = nn_cfg
        self.network = DQN(
            nn_cfg.input_dim,
            nn_cfg.output_dim,
            nn_cfg.features,
        )
        self.loss_fn = self._build_loss(nn_cfg.loss)
        self.optimizer = self._build_optimizer(nn_cfg.optim)
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=nn_cfg.reduce_factor,
                patience=nn_cfg.patience,
            )
        )

    def select_action(self, obs: NDArray):
        Q_s: Tensor = self.network(
            torch.tensor(obs, dtype=torch.float32)
        )
        policy = self.agent_cfg.policy

        if policy.name == "egreedy":
            if (
                random.random() < policy.epsilon
            ):  # If epsilon-greedy we choose a random action
                return torch.randint(
                    0, self.nn_cfg.output_dim, (1,)
                ).item()
            return argmax(Q_s).item()
        elif policy == "softmax":
            probs = softmax(Q_s, policy.temp)
            return torch.multinomial(
                probs, num_samples=1
            ).item()
        else:
            raise ValueError(
                f"{policy} not part of allowed policies. `egreedy` or `softmax`"
            )

    def _build_loss(self, loss: str):
        registry = {
            "L1Loss": nn.L1Loss(),
            "MSELoss": nn.MSELoss(),
            "SmoothL1": nn.SmoothL1Loss(beta=1e-1),
        }
        if loss not in registry:
            raise ValueError(f"Unknown loss: {loss}")
        return registry[loss]

    def _build_optimizer(self, optimizer: str):
        registry = {
            "AdamW": torch.optim.AdamW(
                self.network.parameters(), lr=self.nn_cfg.lr
            ),
            "Adam": torch.optim.Adam(
                self.network.parameters(), lr=self.nn_cfg.lr
            ),
            "RMSprop": torch.optim.RMSprop(
                self.network.parameters(), lr=self.nn_cfg.lr
            ),
        }
        if optimizer not in registry:
            raise ValueError(
                f"Unknown optimizer: {optimizer}"
            )
        return registry[optimizer]

    def update(
        self,
        obs: NDArray,
        action: int,
        reward: int,
        obs_next: int,
        terminated: bool,
        truncated: bool,
    ):
        """Bellman Q-learning update rule"""

        # Tensorise environment variables
        obs_t = torch.tensor(obs, dtype=torch.float)
        obs_next_t = torch.tensor(
            obs_next, dtype=torch.float
        )

        # Getting the Q_values
        Q_sa: Tensor = self.network(obs_t)[action]

        with torch.no_grad():
            target = (
                torch.tensor(reward, dtype=torch.float32)
                if (truncated or terminated)
                else reward
                + self.agent_cfg.gamma
                * torch.max(self.network(obs_next_t))
            )

        loss: Tensor = self.loss_fn(Q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
