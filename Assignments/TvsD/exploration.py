from dataclasses import dataclass


@dataclass
class ExplorationSpec:
    name: str
    maximum_steps: int = 10**6
    # ------------ Agent variables ------------
    # Buffer
    buffer: bool = False
    buffer_size: int = 100

    # Target network
    target_network: bool = False
    update_target: int = 20

    # Policy
    policy: str = (
        "epsilon-greedy"  # ['epsilon-greedy', 'softmax']
    )
    epsilon: float = 0.1
    temperature: float = 1.0

    # NN
    layers: int = 2
    width: int = 64
    lr: float = 5e-4
    batch_size: int = 5

    # ------------ LR Scheduler ------------
    reduce_factor: float = 0.5
    patience: int = 200

    # Evaluation
    n_eval_timesteps: int = 5000
    n_eval_episodes: int = 100


EXPLORATIONS = [
    # Baseline
    ExplorationSpec(
        "BaselineImproved_NN3-128_Batch24_SoftmaxT2.0",
        policy="softmax",
        temperature=2.0,
        batch_size=24,
        layers=3,
        width=128,
    )
    # # Exploration policy
    # ExplorationSpec("Epsilon_0.01", epsilon=0.01),
    # ExplorationSpec("Epsilon_0.05", epsilon=0.05),
    # ExplorationSpec("Epsilon_0.2", epsilon=0.2),
    # ExplorationSpec(
    #     "Softmax_T0.5", policy="softmax", temperature=0.5
    # ),
    # ExplorationSpec(
    #     "Softmax_T1.0", policy="softmax", temperature=1.0
    # ),
    # ExplorationSpec(
    #     "Softmax_T2.0", policy="softmax", temperature=2.0
    # ),
    # # Learning rate
    # ExplorationSpec("LR_1e-3", lr=1e-3),
    # ExplorationSpec(
    #     "LR_5e-4", lr=5e-4
    # ),  # same as baseline, explicit
    # ExplorationSpec("LR_1e-4", lr=1e-4),
    # ExplorationSpec("LR_1e-5", lr=1e-5),
    # # Batch size
    # ExplorationSpec("Batch_1", batch_size=1),
    # ExplorationSpec("Batch_10", batch_size=10),
    # ExplorationSpec("Batch_20", batch_size=20),
    # ExplorationSpec("Batch_64", batch_size=64),
    # # Network architecture
    # ExplorationSpec("Layers_1_W64", layers=1, width=64),
    # ExplorationSpec("Layers_2_W64", layers=2, width=64),
    # ExplorationSpec("Layers_3_W128", layers=3, width=64),
    # # Target network
    # ExplorationSpec("Target_Update10",  target_network=True, update_target=10),
    # ExplorationSpec("Target_Update50",  target_network=True, update_target=50),
    # ExplorationSpec("Target_Update200", target_network=True, update_target=200),
    #
    # # Replay buffer
    # ExplorationSpec("Buffer_100",  buffer=True, buffer_size=100),
    # ExplorationSpec("Buffer_500",  buffer=True, buffer_size=500),
    # ExplorationSpec("Buffer_2000", buffer=True, buffer_size=2000),
    # ExplorationSpec("Buffer_5000", buffer=True, buffer_size=5000),
    # Combined: target network + buffer (closer to canonical DQN)
    #     ExplorationSpec(
    #         "DQN_Canonical",
    #         target_network=True, update_target=100,
    #         buffer=True,         buffer_size=2000,
    #         batch_size=32,
    #         lr=1e-4,
    #         epsilon=0.05,
    #     ),
    #     ExplorationSpec(
    #         "DQN_Canonical_Softmax",
    #         target_network=True, update_target=100,
    #         buffer=True,         buffer_size=2000,
    #         batch_size=32,
    #         lr=1e-4,
    #         policy="softmax",    temperature=1.0,
    #     ),
    # ]
]
