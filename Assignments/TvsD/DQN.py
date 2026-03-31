from gymnasium import Env
import torch
import numpy as np
import copy

from torch import nn
from torch import optim
from numpy.random import rand, choice


class Loss(nn.Module):
    def __init__(
        self,
        policy: str = "epsilon-greedy",
        epsilon: float = 0.1,
        temp: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs
        )  # passes any remaining kwargs to nn.Module
        self.policy = policy
        self.epsilon = epsilon
        self.temp = temp
        assert 0 <= epsilon <= 1


class NaiveLoss(Loss):
    def __init__(
        self,
        policy: str = "epsilon-greedy",
        epsilon: float = 0.1,
        temp: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            policy=policy,
            epsilon=epsilon,
            temp=temp,
            **kwargs,
        )

    def loss_naive(
        self,
        Q_sa: torch.Tensor,
        optimal_Q_sa_next: torch.Tensor,
        r: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        # FIX 2: gamma applied here in the Bellman target, not inside action()
        l = torch.square(
            r + gamma * optimal_Q_sa_next - Q_sa
        )
        return l


class NN(nn.Module):
    def __init__(
        self,
        hidden_layers: int,
        width: int,
        learning_rate: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.h = hidden_layers
        self.w = width
        self.lr = learning_rate
        self.network = self._build_network()

        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=self.lr
        )

    def _build_network(self) -> nn.Sequential:
        layers = [
            nn.Linear(4, self.w),
            nn.ReLU(),
        ]
        for _ in range(self.h):
            layers += [nn.Linear(self.w, self.w), nn.ReLU()]
        layers.append(nn.Linear(self.w, 2))
        return nn.Sequential(*layers)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        x_tensor = torch.tensor(x, dtype=torch.float32)
        return self.network(x_tensor)


class DQNAgent(NN, NaiveLoss):
    def __init__(
        self,
        hidden_layers: int,
        width: int,
        learning_rate: float = 0.01,
        batch_size: int = 1,
        policy: str = "epsilon-greedy",
        epsilon: float = 0.01,
        temp: float = 0.01,
        gamma: float = 0.99,
    ) -> None:
        super().__init__(
            hidden_layers=hidden_layers,
            width=width,
            learning_rate=learning_rate,
            policy=policy,
            epsilon=epsilon,
            temp=temp,
        )

        self.gamma = gamma

        if self.policy not in ["epsilon-greedy", "softmax"]:
            raise ValueError(
                "Policy must be 'epsilon-greedy' or 'softmax'"
            )

    def action(
        self, Q_s: torch.Tensor, optimal: bool = False
    ) -> tuple[int, torch.Tensor]:
        """Choose an action following the current policy (or the greedy policy)."""

        optimal_Q_sa = torch.max(Q_s)
        a = self._policy(Q_s)

        if optimal:
            a = int(torch.argmax(Q_s).item())

        assert a is not None
        return a, optimal_Q_sa

    def eval_Q(self, state: np.ndarray) -> torch.Tensor:
        return self.forward(state)

    def _policy(self, Q_s: torch.Tensor) -> int | None:
        if self.policy == "epsilon-greedy":
            if rand() < self.epsilon:
                return int(choice(range(2)))
            return int(torch.argmax(Q_s).item())

        if self.policy == "softmax":
            probs = (
                torch.softmax(Q_s / self.temp, dim=0)
                .detach()
                .numpy()
            )
            return int(choice(range(2), p=probs))

        return None

    def evaluate(
        self, eval_env: Env, num_episodes: int = 100
    ) -> float:
        returns = []
        for _ in range(num_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            episode_over = False

            while not episode_over:
                with torch.no_grad():  # ← add this
                    Q_s = self.eval_Q(state=obs)
                action, _ = self.action(Q_s, optimal=True)
                obs, reward, terminated, truncated, _ = (
                    eval_env.step(action)
                )
                episode_reward += float(reward)
                episode_over = terminated or truncated

            returns.append(episode_reward)

        return float(np.mean(returns))

    def loss(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> torch.Tensor | None:
        state_t = torch.tensor(state, dtype=torch.float32)
        action_t = torch.tensor(action, dtype=torch.long)
        reward_t = torch.tensor(reward, dtype=torch.float32)
        next_state_t = torch.tensor(
            next_state, dtype=torch.float32
        )
        done_t = torch.tensor(done, dtype=torch.bool)

        # Compute Q(s,a) using the current network
        Q_s = self.eval_Q(
            state_t.numpy()
        )  # eval_Q expects numpy array
        Q_sa = Q_s[int(action_t.item())]

        with torch.no_grad():  # target should be a fixed value, not part of the graph
            Q_s_next = self.eval_Q(next_state_t.numpy())
            optimal_Q_sa_next = (
                torch.max(Q_s_next)
                if not done_t.item()
                else torch.tensor(0.0, dtype=torch.float32)
            )

        l = self.loss_naive(
            Q_sa=Q_sa,
            optimal_Q_sa_next=optimal_Q_sa_next,
            r=reward_t,
            gamma=self.gamma,
        )

        return l
