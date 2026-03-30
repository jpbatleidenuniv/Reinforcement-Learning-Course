from dataclasses import dataclass

@dataclass
class args():
    # ------------ Experimental setup variables ------------
    num_eval_episodes: int = 4000
    recording_interval: int = 400

    # ------------ Agent variables ------------
    # Buffer
    buffer: bool = False
    buffer_size: int = 100

    # Target network
    target_network: bool = False
    update_target: int = 20

    # Policy
    policy: str = 'epsilon-greedy' # ['epsilon-greedy', 'softmax']
    epsilon: float = 0.01
    temperature: float = 1.0

    # NN
    layers: int = 2
    width: int = 64
    output_len: int = 2
    input_len: int = 2
    lr: float = 0.000005

    # ------------ LR Scheduler ------------
    reduce_factor: float = 0.5
    patience: int = 400