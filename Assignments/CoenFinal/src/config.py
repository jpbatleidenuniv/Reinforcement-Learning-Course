from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Sequence


@dataclass(frozen=True)
class RunConfig:
    name: str
    n_eval_episodes: int = 10
    n_steps: float = 250_000


@dataclass(frozen=True)
class NNConfig:
    input_dim: int
    output_dim: int
    features: Sequence[int]
    policy_lr: float
    value_lr: float
    optim: str
    loss: str = "MSE"
    lamb: float = 0.9


@dataclass(frozen=True)
class AgentConfig:
    nn_cfg: NNConfig
    gamma: float = 0.99
    n_steps: int = 10
    epsilon: float = 0.2
    k: int = 10


@dataclass(frozen=True)
class Config:
    run: RunConfig
    nn: NNConfig
    agent: AgentConfig


CONFIGS = [
    # RunConfig("Best_Combined",
    #     policy="softmax",       # <-- replace with your findings
    #     temperature=1.0,
    #     layers=2,
    #     width=64,
    #     lr=1e-3,
    #     batch_size=32,
    # ),
    # ================================================================
    #     # STUDY 2: Stabilization techniques (use best hyperparams above)
    #     # ================================================================
    #     RunConfig("Naive",
    #         policy="softmax", temperature=1.0,  # <-- replace with best
    #         layers=2, width=64, lr=1e-3, batch_size=64,
    #     ),
    #     RunConfig("TargetNetwork",
    #         policy="softmax", temperature=1.0,
    #         layers=2, width=64, lr=1e-3, batch_size=64,
    #         target_network=True, update_target=100,
    #     ),
    #     RunConfig("ExperienceReplay",
    #         policy="softmax", temperature=1.0,
    #         layers=2, width=64, lr=1e-3, batch_size=64,
    #         buffer=True, buffer_size=100000, min_buffer_size=1000,
    #     ),
    #     RunConfig("TargetNetwork_ExperienceReplay",
    #         policy="softmax", temperature=1.0,
    #         layers=2, width=64, lr=1e-3, batch_size=64,
    #         target_network=True, update_target=100,
    #         buffer=True, buffer_size=100000, min_buffer_size=1000,
    #     ),
]

N_REPETITIONS = 5
