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
            output_dim: int,
            features: Sequence[int],
    ):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, features[0]), nn.ReLU())
        self.body = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(features[i], features[i + 1]),
                    nn.ReLU()
                )
                for i in range(len(features) - 1)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(features[-1], output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        for layer in self.body:
            x = layer(x)
        return self.output_layer(x)


class PolicyAgent:
    def __init__(self, agent_cfg: AgentConfig, nn_cfg: NNConfig, advantage: bool = False) -> None:
        self.agent_cfg = agent_cfg
        self.nn_cfg = nn_cfg
        self.policy = PINetwork(
            nn_cfg.input_dim,
            nn_cfg.output_dim,
            nn_cfg.features,
        )

        self.advantage = advantage
        self.optimizer = self._build_optimizer(nn_cfg.optim)
        # Define arrays of rollout monte carlo
        self.log_probs: list[Tensor] = []
        self.rewards: list[float] = []

    def select_action(self, obs: NDArray) -> tuple[int, Tensor, Tensor]:
        """Stochastic action selection — used during training."""
        pi_s: Tensor = self.policy(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        ).squeeze(0)
        dist = torch.distributions.Categorical(pi_s)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob, pi_s

    def select_greedy_action(self, obs: NDArray) -> int:
        """Greedy (deterministic) action selection — used during evaluation."""
        with torch.no_grad():
            pi_s: Tensor = self.policy(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0)
        return int(torch.argmax(pi_s).item())

    def _build_optimizer(self, optimizer: str):
        registry = {
            "AdamW": torch.optim.AdamW(self.policy.parameters(), lr=self.nn_cfg.policy_lr),
            "Adam": torch.optim.Adam(self.policy.parameters(), lr=self.nn_cfg.policy_lr),
            "RMSprop": torch.optim.RMSprop(self.policy.parameters(), lr=self.nn_cfg.policy_lr),
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

    def update(self, objectives: None | Tensor = None):
        self.optimizer.zero_grad()
        T = len(self.rewards)

        if objectives is None:
            objectives = self._returns  # T targets, this is G_t

        else:
            assert len(objectives) == T, (
                f"The length of G_t is not the same as the episode length. "
                f"This is in the 'update' method in PolicyAgent and G_t has a shape of {objectives.shape}"
            )
            objectives = objectives

        if not self.advantage:
            objectives = (objectives - objectives.mean()) / (objectives.std() + 1e-8)
        assert T == len(objectives), f"T = {T}, len(G_T) = {len(objectives)}"
        log_probs = torch.stack(self.log_probs)
        loss = -1 * (log_probs * objectives).sum()
        
        normalizer = torch.tensor(len(log_probs)).detach()
        loss /= normalizer
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        info = {"loss": loss.item(), "step": T, "episode_return": sum(self.rewards)}

        # We wipe the episode information for the next episode
        self.log_probs.clear()
        self.rewards.clear()

        return info


class ValueAgent:
    def __init__(self, agent_cfg: AgentConfig, nn_cfg: NNConfig, advantage=False) -> None:
        self.agent_cfg = agent_cfg
        self.nn_cfg = nn_cfg

        self.value = ValueNetwork(
            nn_cfg.input_dim,
            output_dim=1,
            features=nn_cfg.features,
        )

        self.optimizer = self._build_optimizer(nn_cfg.optim)
        self.loss = self._loss(nn_cfg.loss)
        self.n_steps = agent_cfg.n_steps

        # Define arrays of rollout monte carlo
        self.V_s: list[Tensor] = []
        self.rewards: list[float] = []
        self._g_t: list[Tensor] = []

        self.advantage = advantage

    def _build_optimizer(self, optimizer: str):
        registry = {
            "AdamW": torch.optim.AdamW(self.value.parameters(), lr=self.nn_cfg.value_lr),
            "Adam": torch.optim.Adam(self.value.parameters(), lr=self.nn_cfg.value_lr),
            "RMSprop": torch.optim.RMSprop(self.value.parameters(), lr=self.nn_cfg.value_lr),
        }
        if optimizer not in registry:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        return registry[optimizer]

    def _loss(self, loss: str):
        registry = {
            "MSE": torch.nn.MSELoss(),
            "Huber": torch.nn.HuberLoss(reduction='mean', delta=self.agent_cfg.n_steps)
        }
        if loss not in registry:
            raise ValueError(f"Unknown loss function: {loss}")
        return registry[loss]

    def values(self, obs: Tensor) -> Tensor:
        # squeeze() removes both the batch dim and the output_dim=1 dim -> scalar tensor
        Q_s = self.value.forward(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        ).squeeze()
        return Q_s

    @property
    def G_t(self):
        T = len(self.rewards)
        gamma = self.agent_cfg.gamma

        gammas = torch.pow(
            torch.tensor(gamma, dtype=torch.float32),
            torch.arange(self.n_steps, dtype=torch.float32),
        )

        bootstrap_gamma = gamma ** self.n_steps

        g_t: list[Tensor] = []

        assert len(self.V_s) == T, (
            f"V_s has length {len(self.V_s)}, while T has length {T}"
        )

        for k in range(T):
            if k + self.n_steps < T:

                rewards = torch.tensor(
                    self.rewards[k: k + self.n_steps], dtype=torch.float32
                )
                discounted_r = gammas * rewards                              
                discounted_v = bootstrap_gamma * self.V_s[k + self.n_steps]  
                g = discounted_r.sum() + discounted_v

            else:
                # Tail: fewer than n steps remain — use available rewards only, no bootstrap
                rewards = torch.tensor(self.rewards[k:], dtype=torch.float32)
                discounted_r = gammas[:len(rewards)] * rewards
                g = torch.sum(discounted_r, dtype=torch.float32)

            if self.advantage:
                a = g - self.V_s[k].detach()
                g_t.append(a.detach())
            else:
                g_t.append(g.detach())

        self._g_t = g_t
        return torch.stack(g_t)

    def update(self):
        target = torch.stack(self._g_t).detach()
        target = (target - target.mean()) / (target.std() + 1e-8)
        pred = torch.stack(self.V_s)

        l = self.loss(pred, target)
        self.optimizer.zero_grad()
        l.backward()

        self.optimizer.step()

        info = {"loss": l.detach().item()}

        # We wipe the episode information for the next episode
        self.V_s = []
        self.rewards = []
        self._g_t = []

        return info