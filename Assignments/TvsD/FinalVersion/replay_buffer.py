import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer using a FIFO (First-In First-Out) eviction policy.
    """

    def __init__(self, capacity: int, obs_dim: int):
        """
        Args:
            capacity: Maximum number of transitions to store.
            obs_dim:  Dimensionality of a single observation.
        """
        self.capacity = capacity
        self.obs_dim = obs_dim

        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)  # Stored as float for easy TD masking

        self.ptr  = 0  # Points to the next write position (wraps around at capacity)
        self.size = 0  # Current number of valid transitions stored

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """
        Add a single transition to the buffer.

        If the buffer is full, the oldest transition is silently overwritten (FIFO).
        """
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)

        # Advance the write pointer, wrapping around to overwrite old data when full
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Sample a random batch of transitions (uniform, with replacement).

        """
        # Only sample from positions that have been written at least once
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size
