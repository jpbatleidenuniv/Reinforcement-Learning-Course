from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class RunConfig:
    name: str
    max_steps: int = 500_000  # Total max steps for a run
    n_episodes: int = 500  # Number of episodes
    evaluation_interval: int = (
        10  # At every {evaluation_interval} episodes, evaluation will happen
    )
    n_eval_episodes: int = 20  # Evaluation will average over 20 episods


@dataclass(frozen=True)
class NNConfig:
    input_dim: int
    output_dim: int
    features: Sequence[int]
    lr: float
    optim: str


@dataclass(frozen=True)
class AgentConfig:
    nn_cfg: NNConfig
    gamma: float = 0.99


@dataclass(frozen=True)
class Config:
    run: RunConfig
    pi_nn: NNConfig
    v_nn: NNConfig
    agent: AgentConfig
