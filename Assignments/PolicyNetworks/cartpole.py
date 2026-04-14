import gymnasium as gym
import numpy as np


class CartPoleEnv:
    """
    Wrapper function that can make a vectorized, or singular Cartpole environment

    All outputs are always batched:
        obs:     (num_envs, obs_dim)
        rewards: (num_envs,)
        dones:   (num_envs,)

    With num_envs=1 a standard gym.make environment is used under the hood.
    With num_envs>1 a gym.make_vec environment is used, which runs envs in parallel.
    """

    def __init__(self, num_envs: int = 1, seed: int = 0, render_mode: str | None = None):
        self.num_envs = num_envs
        self.seed = seed
        self.render_mode = render_mode
        self.is_vectorized = num_envs > 1

        if self.is_vectorized:
            # Vectorized env: all num_envs instances step simultaneously
            self.env = gym.make_vec("CartPole-v1", num_envs=num_envs)
            self._obs_dim = self.env.single_observation_space.shape[0]
            self._action_dim = self.env.single_action_space.n
        else:
            # Single env: wrapped to always return batched outputs
            self.env = gym.make("CartPole-v1", render_mode=render_mode)
            self._obs_dim = self.env.observation_space.shape[0]
            self._action_dim = self.env.action_space.n

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def n_envs(self) -> int:
        return self.num_envs

    def reset(self):
        """
        Reset all environment and return initial observations
        """
        obs, info = self.env.reset(seed=self.seed)

        # Add batch dimension for the single-env case to keep shapes consistent
        if not self.is_vectorized:
            obs = np.expand_dims(obs, axis=0)

        return obs, info

    def step(self, actions):
        """
        Step all environments forward by one timestep.
        """
        if not self.is_vectorized:
            # Gymnasium's single env expects a plain int, not an array
            actions = int(np.asarray(actions).reshape(-1)[0])

        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        if not self.is_vectorized:
            # Wrap scalars/arrays in a batch dimension to keep the interface uniform
            obs = np.expand_dims(obs, axis=0)
            rewards = np.asarray([rewards], dtype=np.float32)
            terminated = np.asarray([terminated], dtype=bool)
            truncated = np.asarray([truncated], dtype=bool)

        # Combine terminated and truncated into a single done signal
        dones = np.logical_or(terminated, truncated)
        return obs, rewards, dones, infos

    def sample_random_actions(self):
        """Sample random actions for all environments."""
        if self.is_vectorized:
            return self.env.action_space.sample()
        return np.asarray([self.env.action_space.sample()], dtype=np.int64)

    def close(self):
        self.env.close()
