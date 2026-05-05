from collections.abc import Sequence
from torch import Tensor
from numpy.typing import NDArray
from config import AgentConfig, NNConfig
import torch.nn as nn
import torch
import numpy as np
from gymnasium import Env


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


class VNetwork(nn.Module):
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
        self.output_layer = nn.Linear(features[-1], output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        for layer in self.body:
            x = layer(x)
        return self.output_layer(x)


class BaseAgent:
    def __init__(self, agent_cfg: AgentConfig, policy_network_cfg: NNConfig) -> None:
        self.agent_cfg = agent_cfg
        self.policy_network_cfg = policy_network_cfg
        self.policy = PINetwork(
            policy_network_cfg.input_dim,
            policy_network_cfg.output_dim,
            policy_network_cfg.features,
        )
        self.optimizer = None

    def select_action(self, obs: NDArray) -> tuple[int, Tensor]:
        # First we make predictions fro policy at state 'obs'
        pi_s: Tensor = self.policy(
            torch.tensor(
                np.array(obs, dtype=np.float32), dtype=torch.float32
            ).unsqueeze(0)
        ).squeeze(0)  # Softmaxed dist of 2 possible actions
        dist = torch.distributions.Categorical(pi_s)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob

    def _build_optimizer(self, optimizer: str, parameters, lr):
        factories = {
            "AdamW": torch.optim.AdamW,
            "Adam": torch.optim.Adam,
            "RMSprop": torch.optim.RMSprop,
        }
        if optimizer not in factories:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        return factories[optimizer](parameters, lr=lr)

    def update(self):
        raise NotImplementedError(
            f"{self.__repr__()} is not intended for use. Use the children Agents"
        )


class ReinforceAgent(BaseAgent):
    def __init__(self, agent_cfg: AgentConfig, policy_network_cfg: NNConfig) -> None:
        super().__init__(agent_cfg, policy_network_cfg)
        self.optimizer = self._build_optimizer(
            "Adam", self.policy.parameters(), lr=policy_network_cfg.lr
        )
        # Define arrays of rollout monte carlo
        self.log_probs: list[Tensor] = []
        self.rewards: list[float] = []
        self.observations = []

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

    @property
    def _reinforce_loss(self):
        """The REINFORCE loss reads:
        - log(pi_s) * G_t
        """
        log_probs = torch.stack(self.log_probs)  # shape T, 4
        G_t = (self._returns - self._returns.mean()) / (self._returns.std() + 1e-8)
        return -1 * (log_probs * G_t).sum()

    def update(self):
        """Update rule for the REINFORCE agent, clears registries at the end of each episode. Uses returns as objectives"""
        self.optimizer.zero_grad()
        T = len(self.rewards)  # capture here
        loss = self._reinforce_loss
        loss.backward()
        self.optimizer.step()

        info = {
            "loss": loss.item(),
            "step": T,
            "episode_return": sum(self.rewards),
        }
        self.info = info

        self.log_probs.clear()
        self.rewards.clear()
        self.observations.clear()

        return info

    def evaluate(self, env: Env, n_episodes: int) -> float:
        """Evaluate the agent on a separate environment, returns mean return over n_episodes"""
        total_returns = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            truncated, terminated = False, False
            episode_return = 0.0
            steps = 0
            while not (truncated or terminated):
                with torch.no_grad():
                    action, _ = self.select_action(obs)
                obs, r, terminated, truncated, _ = env.step(action)
                episode_return += float(r)
                steps += 1
            total_returns.append(episode_return)

        # Select action populates the experience
        # This should not happen for evaluation
        self.log_probs.clear()
        self.rewards.clear()
        self.observations.clear()
        return float(np.mean(total_returns))


class ACAgent(ReinforceAgent):
    def __init__(
        self,
        agent_cfg: AgentConfig,
        policy_network_cfg: NNConfig,
        value_network_cfg: NNConfig,
    ) -> None:
        super().__init__(agent_cfg, policy_network_cfg)
        self.value_network_cfg = value_network_cfg
        self.V = VNetwork(
            value_network_cfg.input_dim,
            value_network_cfg.output_dim,
            value_network_cfg.features,
        )
        self.critic_optimizer = self._build_optimizer(
            "Adam", self.V.parameters(), self.value_network_cfg.lr
        )

    def select_action(self, obs: NDArray) -> tuple[int, Tensor]:
        """Same action taking policy. But now we track the state for updating the value network"""
        action, log_prob = super().select_action(obs)
        self.observations.append(obs)
        return action, log_prob

    @property
    def _critic_loss(self) -> Tensor:
        """The loss for the critic (Value network) is the MSE between the G_t
        V(s_t), representing the difference between predicted (expected) and the real cumulative return under policy
        """
        loss_fn = nn.MSELoss()

        # Compute V(s) via network
        obs_tensor = torch.tensor(
            np.array(self.observations, dtype=np.float32), dtype=torch.float32
        )
        V_pred = self.V(obs_tensor).squeeze()  # actual network prediction
        return loss_fn(V_pred, self._returns.detach())  # (prediction, target)

    def update(self):
        # First we update the critic
        self.critic_optimizer.zero_grad()
        loss_critic = self._critic_loss
        loss_critic.backward()
        self.critic_optimizer.step()

        # Then we update the actor using the same as from REINFORCE
        info = super().update()  # This clears the registry for the next run, so we need to perform this after self._critic_loss to have available rewards
        info["value_loss"] = loss_critic.item()
        return info

    def evaluate(self, env, n_episodes):
        result = super().evaluate(env, n_episodes)
        self.observations.clear()  # clean up side effects from select_action
        return result


class A2CAgent(ACAgent):
    def __init__(
        self,
        agent_cfg: AgentConfig,
        policy_network_cfg: NNConfig,
        value_network_cfg: NNConfig,
    ) -> None:
        super().__init__(agent_cfg, policy_network_cfg, value_network_cfg)

    @property
    def _actor_loss(self):
        """
        The same as the REINFORCE loss but uses G_t - V(s) as the objective, this is known as the advantage. This is objective the actor compares on
        """
        log_probs = torch.stack(self.log_probs)  # shape T, 4
        G_t = (self._returns - self._returns.mean()) / (self._returns.std() + 1e-8)

        # Compute V(s) via network
        obs_tensor = torch.tensor(
            np.array(self.observations, dtype=np.float32), dtype=torch.float32
        )
        V_pred = self.V(obs_tensor).squeeze()  # actual network prediction
        advantage = G_t - V_pred.detach()
        return -1 * (log_probs * advantage).sum()

    def update(self):
        """Update rule for the Advantage actor critic. Here the policy network
        is updated using the outputs from the value network. The loss reads
        - log(pi_s)(G_t - V(s))"""
        # First we update the critic
        self.critic_optimizer.zero_grad()
        T = len(self._returns)
        loss_critic = self._critic_loss
        loss_critic.backward()
        self.critic_optimizer.step()

        # Updating the actor
        self.optimizer.zero_grad()
        loss_actor = self._actor_loss
        loss_actor.backward()
        self.optimizer.step()

        info = {
            "loss": loss_actor.item(),
            "value_loss": loss_critic.item(),
            "step": T,
            "episode_return": sum(self.rewards),
        }
        self.info = info

        self.log_probs.clear()
        self.rewards.clear()
        self.observations.clear()
        return info
