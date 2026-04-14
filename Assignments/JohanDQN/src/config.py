from __future__ import annotations
from dataclasses import dataclass, field
from collections.abc import Sequence


@dataclass(frozen=True)
class RunConfig:
    name: str
    max_steps: int = 10**6
    n_eval_timesteps: int = 5_000
    n_eval_episodes: int = 100


@dataclass(frozen=True)
class NNConfig:
    input_dim: int
    output_dim: int
    features: Sequence[int] = field(
        default_factory=lambda: [64, 128, 64]
    )
    lr: float = 1e-4
    loss: str = "MSELoss"  # must match _build_loss registry
    optim: str = (
        "AdamW"  # must match _build_optimizer registry
    )
    reduce_factor: float = 0.5
    patience: int = 1_000
    batch_size: int = 32


@dataclass(frozen=True)
class PolicyConfig:
    name: str = "egreedy"
    epsilon: float = 0.2
    temp: float = 1.0


@dataclass(frozen=True)
class AgentConfig:
    policy: PolicyConfig = field(
        default_factory=PolicyConfig
    )
    gamma: float = 0.99


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
