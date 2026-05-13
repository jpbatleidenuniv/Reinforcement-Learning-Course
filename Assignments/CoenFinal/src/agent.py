from collections.abc import Sequence
from torch import Tensor
from numpy.typing import NDArray
from config import AgentConfig, NNConfig
from copy import deepcopy

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
        self.body = nn.ModuleList([
                                    nn.Sequential(
                                                  nn.Linear(features[i], features[i + 1]),
                                                  nn.ReLU()
                                                 )
                                    for i in range(len(features) - 1)
                                 ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(features[-1], output_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        for layer in self.body:
            x = layer(x)
        return self.output_layer(x)


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
        self.ratios: list[Tensor] = []
        self.rewards: list[float] = []
        self.old_policy  = deepcopy(self.policy)

    def _update_od_policy(self):
        self.old_policy = deepcopy(self.policy)

    def select_action(self, obs: NDArray) -> tuple[int, Tensor]:
        """Stochastic action selection, used during training."""

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        pi_s: Tensor = self.policy(obs_tensor).squeeze(0)
        
        old_pi_s: Tensor = self.old_policy(obs_tensor).squeeze(0)
        
        dist = torch.distributions.Categorical(old_pi_s)
        action = dist.sample()

        pi_sa = pi_s[action]
        old_pi_sa = old_pi_s[action].detach()
        r_sa = pi_sa / old_pi_sa
        return int(action.item()), r_sa
    

    def select_greedy_action(self, obs: NDArray) -> int:
        """Greedy (deterministic) action selection. Used during evaluation."""
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
        returns: Tensor = torch.cumsum(discounted.flip(0), dim=0).flip(0)
        return returns

    def update(self, advantage: Tensor):
        self.optimizer.zero_grad()
        T = len(self.rewards)

        # advantage = advantage.detach()

        assert len(advantage) == T, (
            f"The length of G_t is not the same as the episode length. "
            f"This is in the 'update' method in PolicyAgent and G_t has a shape of {advantage.shape}"
        )

        # Update old to new policy for next iteration
        self.old_policy = self.policy


        ratios = torch.stack(self.ratios)
        min_clip = torch.tensor((1-self.agent_cfg.epsilon), dtype=torch.float32)
        max_clip = torch.tensor((1+self.agent_cfg.epsilon), dtype=torch.float32)
        clipped_ratios = torch.clip(ratios, min_clip, max_clip)
        min_advantages = -torch.min(ratios*advantage, clipped_ratios*advantage)
        assert len(min_advantages) == len(ratios), f"The length of the clipped and minimized ratios * advantages is not the same as the lenth of the ratios. Should be: len(r * A) : {len(min_advantages)} | len(ratios) : {len(ratios)}"
        
        loss = torch.mean(-torch.min(ratios * advantage, clipped_ratios * advantage))
        loss.backward()
        self.optimizer.step()

        info = {"loss": loss.item(), "step": T, "episode_return": sum(self.rewards)}

        # We wipe the episode information for the next episode
        self.ratios.clear()
        self.rewards.clear()

        return info


class ValueAgent:
    def __init__(self, agent_cfg: AgentConfig, nn_cfg: NNConfig) -> None:
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
        self._a_t: list[Tensor] = []


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
        Q_s = self.value.forward(
            torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        ).squeeze()
        return Q_s

    def advantage(self):
        T = len(self.rewards)
        gamma = self.agent_cfg.gamma

        # gammas[i] = y^i, length n_steps. used to discount the n reward steps
        gammas = torch.pow(
            torch.tensor(gamma, dtype=torch.float32),
            torch.arange(self.n_steps, dtype=torch.float32),
        )
        # Scalar bootstrap factor y^n_steps (one step beyond the n reward steps)
        bootstrap_gamma = gamma ** self.n_steps

        # List with advantages
        a_t: list[Tensor] = []

        assert len(self.V_s) == T, (
            f"V_s has length {len(self.V_s)}, while T has length {T}"
        )
        

        """eed to doublecheck on how to compute the advantage. Right now I just take the rewards, but perhaps it might be better to take the estimated Q_sa
        of the value network. Might not be needed though, so think about it"""
        for k in range(T): 
            if k + self.n_steps < T:

                rewards = torch.tensor(
                    self.rewards[k: k + self.n_steps], dtype=torch.float32
                )
                discounted_r = gammas * rewards                              # y^0*r_k … y^{n-1}*r_{k+n-1}
                discounted_v = bootstrap_gamma * self.V_s[k + self.n_steps]  # y^n * V(s_{k+n})
                g = discounted_r.sum() + discounted_v

            else:
                # Tail: fewer than n steps remain. Use available rewards only, no bootstrap
                rewards = torch.tensor(self.rewards[k:], dtype=torch.float32)
                discounted_r = gammas[:len(rewards)] * rewards
                g = torch.sum(discounted_r, dtype=torch.float32)

            # Advantage
            a = g - self.V_s[k].detach()
            a_t.append(a.detach())

        self._a_t = a_t
        return torch.stack(a_t)

    def update(self):
        target = torch.stack(self._a_t)
        pred = torch.stack(self.V_s)

        l = self.loss(pred, target)/len(pred)
        self.optimizer.zero_grad()
        l.backward()

        self.optimizer.step()

        info = {"loss": l.detach().item()}

        # We wipe the episode information for the next episode
        self.V_s = []
        self.rewards = []
        self._g_t = []

        return info