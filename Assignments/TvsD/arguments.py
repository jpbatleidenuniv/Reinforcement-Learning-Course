from dataclasses import dataclass


@dataclass
class ExplorationSpec:
    name: str
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


EXPLORATIONS = [ExplorationSpec("")]
