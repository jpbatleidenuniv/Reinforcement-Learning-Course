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


class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        features: Sequence[int],
    ):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, features[0]), nn.ReLU())

        self.body = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(features[i], features[i + 1]), nn.ReLU())
                for i in range(len(features) - 1)
            ]
        )

        self.output_layer = nn.Linear(features[-1], 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        for layer in self.body:
            x = layer(x)
        return self.output_layer(x).squeeze(-1)


class PolicyAgent:
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

    def select_action(self, obs: NDArray) -> tuple[int, Tensor, Tensor]:
        # First we make predictions fro policy at state 'obs'
        pi_s: Tensor = self.policy(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        ).squeeze(
            0
        )  # Softmaxed dist of 2 possible actions
        dist = torch.distributions.Categorical(pi_s)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob, pi_s

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
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        gammas = torch.pow(
            torch.tensor(self.agent_cfg.gamma, dtype=torch.float32),
            torch.arange(T, dtype=torch.float32),
        )
        discounted = gammas * rewards

        # Discounted cumulative returns from t=0 to t=T (episode length)
        returns: Tensor = torch.cumsum(discounted.flip(0), dim=0).flip(0) / gammas
        return returns

    def update(self, objectives: None | Tensor = None):
        self.optimizer.zero_grad()
        T = len(self.rewards)

        if objectives is None:
            objectives = self._returns  # T targets, this is G_t

        else:
            assert (
                len(objectives) == T
            ), f"The length of G_t is not the same as the episode length. This is in the 'update' method in PolicyAgent and G_t has a shape of {objectives.shape}"
            objectives = objectives

        # we do not want standardization
        # objectives = (objectives - objectives.mean()) / (objectives.std() + 1e-8)
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


class ValueAgent:
    def __init__(
        self, agent_cfg: AgentConfig, nn_cfg: NNConfig, advantage=False
    ) -> None:
        self.agent_cfg = agent_cfg
        self.nn_cfg = nn_cfg

        self.value = ValueNetwork(
            nn_cfg.input_dim,
            nn_cfg.features,
        )

        self.optimizer = self._build_optimizer(nn_cfg.optim)
        self.loss = nn.MSELoss()
        self.n_steps = agent_cfg.n_steps
        self.advantage = advantage

        self.V_s: list[Tensor] = []
        self.rewards: list[float] = []
        self._g_t: list[Tensor] = []

    def _build_optimizer(self, optimizer: str):
        registry = {
            "AdamW": torch.optim.AdamW(self.value.parameters(), lr=self.nn_cfg.lr),
            "Adam": torch.optim.Adam(self.value.parameters(), lr=self.nn_cfg.lr),
            "RMSprop": torch.optim.RMSprop(self.value.parameters(), lr=self.nn_cfg.lr),
        }
        if optimizer not in registry:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        return registry[optimizer]

    def values(self, obs: NDArray) -> Tensor:
        V_s = self.value(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)

        return V_s

    @property
    def G_t(self):
        T = len(self.rewards)
        g_t: list[Tensor] = []

        assert (
            len(self.V_s) == T
        ), f"V_s has length {len(self.V_s)}, while rewards has length {T}"

        for k in range(T):
            G = torch.tensor(0.0, dtype=torch.float32)

            max_n = min(self.n_steps, T - k)

            for i in range(max_n):
                G = G + (self.agent_cfg.gamma**i) * self.rewards[k + i]

            if k + self.n_steps < T:
                G = (
                    G
                    + (self.agent_cfg.gamma**self.n_steps)
                    * self.V_s[k + self.n_steps].detach()
                )

            g_t.append(G.detach())

        self._g_t = g_t

        targets = torch.stack(g_t)

        if self.advantage:
            values = torch.stack(self.V_s).detach()
            return targets - values

        return targets

    def update(self):
        target = torch.stack(self._g_t).detach()
        pred = torch.stack(self.V_s)

        loss = self.loss(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        info = {"loss": loss.detach().item()}

        self.V_s.clear()
        self.rewards.clear()
        self._g_t.clear()

        return info
