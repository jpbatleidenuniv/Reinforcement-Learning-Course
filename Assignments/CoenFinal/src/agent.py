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


class PolicyAgent:
    def __init__(self, agent_cfg: AgentConfig, nn_cfg: NNConfig) -> None:
        self.agent_cfg = agent_cfg
        self.nn_cfg = nn_cfg
        self.policy = PINetwork(
            nn_cfg.input_dim,
            nn_cfg.output_dim,
            nn_cfg.features,
        )
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=nn_cfg.policy_lr)
        self.old_policy = deepcopy(self.policy)

    def select_action(self, obs: list[NDArray]) -> tuple[int, Tensor]:
        """Stochastic action selection, used during training."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        old_pi_s: Tensor = self.old_policy(obs_tensor).squeeze(0)
        dist = torch.distributions.Categorical(old_pi_s)
        action = dist.sample()
        old_pi_sa = old_pi_s[action].detach()
        return int(action.item()), old_pi_sa

    def probability_new_policy(self, obs: list[Tensor], action: list[int]) -> Tensor:
        """Returns pi_sa for the network currently being optimised."""
        obs_tensor    = torch.stack(obs)                                    # [T, input_dim]
        pi_s: Tensor  = self.policy(obs_tensor)                             # [T, output_dim]
        action_tensor = torch.tensor(action, dtype=torch.long)
        return pi_s[torch.arange(pi_s.size(0)), action_tensor]             # [T]

    def select_greedy_action(self, obs: NDArray) -> int:
        """Greedy action selection. Used during evaluation."""
        with torch.no_grad():
            pi_s: Tensor = self.policy(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            ).squeeze(0)
        return int(torch.argmax(pi_s).item())

    def update_old_policy(self):
        """Copies network being optimised into the old (behaviour) network."""
        self.old_policy.load_state_dict(self.policy.state_dict())

    def update(self, advantage: Tensor, ratios: Tensor):
        self.optimizer.zero_grad()
        T = len(ratios)

        advantage = advantage.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        min_clip      = torch.tensor(1 - self.agent_cfg.epsilon, dtype=torch.float32)
        max_clip      = torch.tensor(1 + self.agent_cfg.epsilon, dtype=torch.float32)
        clipped_ratios = torch.clip(ratios, min_clip, max_clip)

        loss = torch.mean(-torch.min(ratios * advantage, clipped_ratios * advantage))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return {"loss": loss.item(), "step": T}


############################################## VALUE AGENT ###########################################################


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, features: Sequence[int]):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, features[0]), nn.ReLU())
        self.body = nn.ModuleList([
            nn.Sequential(nn.Linear(features[i], features[i + 1]), nn.ReLU())
            for i in range(len(features) - 1)
        ])
        self.output_layer = nn.Linear(features[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        for layer in self.body:
            x = layer(x)
        return self.output_layer(x)


class ValueAgent:
    def __init__(self, agent_cfg: AgentConfig, nn_cfg: NNConfig) -> None:
        self.agent_cfg = agent_cfg
        self.nn_cfg    = nn_cfg
        self.value     = ValueNetwork(nn_cfg.input_dim, output_dim=1, features=nn_cfg.features)
        self.optimizer = torch.optim.AdamW(self.value.parameters(), lr=self.nn_cfg.value_lr)
        self.loss      = torch.nn.SmoothL1Loss(beta=1.0)
        self.n_steps   = agent_cfg.n_steps

    def values(self, obs) -> Tensor:
        return self.value(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze()

    def update(self, v_s: Tensor, r, done):
        """
        v_s  : [T+1] tensor WITH gradients.
               v_s[:-1] are the state-value predictions V(s_0)...V(s_{T-1}).
               v_s[-1]  is the terminal bootstrap value (zeroed by caller if
               the episode ended naturally).
        r    : list[float] of length T — rewards collected during rollout.
        done : list[bool]  of length T — episode-end flags.
        """

        V_preds = v_s[:-1]      # [T]  — keeps grad, used in loss
        v_s_d   = v_s.detach()  # [T+1] — no grad, used only inside target computation

        gamma   = self.agent_cfg.gamma
        n_steps = self.agent_cfg.n_steps
        T       = len(r)

        r    = torch.tensor(r,    dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)

        targets = torch.zeros(T, dtype=torch.float32)
        for t in range(T):
            R       = 0.0
            horizon = t  # will track the true end of the n-step window
            for k in range(t, min(t + n_steps, T)):
                R      += (gamma ** (k - t)) * r[k].item()
                horizon = k + 1
                if done[k]:
                    break           # episode boundary — no bootstrap beyond here
            else:
                # n-step window completed without hitting a done — bootstrap
                R += (gamma ** (horizon - t)) * v_s_d[horizon].item()
            targets[t] = R

        loss = self.loss(V_preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {"loss": loss.item()}