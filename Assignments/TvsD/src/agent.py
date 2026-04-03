from collections.abc import Sequence
from torch import Tensor
from torchsummary import summary
from numpy.typing import NDArray
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
    z = x - torch.max(
        x
    )  # substract max to prevent overflow of softmax
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


if __name__ == "__main__":
    model = DQN(4, 2, [12, 24, 48, 24, 12])
    summary(model, input_size=(1, 4))


class Agent:
    def __init__(
        self,
        lr: float,
        gamma: float,
        policy: str,
        epsilon: float,
        temp: float,
        input_dim=4,
        output_dim=2,
        features: Sequence[int] = [32, 64, 32],
    ) -> None:
        self.network = DQN(input_dim, output_dim, features)
        self.lr = lr
        self.gamma = gamma
        self.policy = policy
        self.epsilon = epsilon
        self.temp = temp
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.features = features

    def select_action(self, obs: NDArray):
        Q_s: Tensor = self.network(
            torch.tensor(obs, dtype=torch.float32)
        )
        print("The network gives us", Q_s)
        policy = self.policy

        if policy == "egreedy":
            if (
                random.random() < self.epsilon
            ):  # If epsilon-greedy we choose a random action
                return torch.randint(
                    0, self.output_dim, (1,)
                )
            return argmax(Q_s)
        elif policy == "softmax":
            return softmax(Q_s, self.temp)
        else:
            raise ValueError(
                f"{policy} not part of allowed policies. `egreedy` or `softmax`"
            )

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

        target = (
            reward
            if (truncated or terminated)
            else reward
            + self.gamma
            * torch.max(self.network(obs_next_t))
        )


if __name__ == "__main__":
    agent = Agent(
        lr=5e-4,
        gamma=0.99,
        policy="egreedy",
        epsilon=0.3,
        temp=2.0,
    )

    obs = np.random.randn(4)
    print(agent.select_action(obs))
