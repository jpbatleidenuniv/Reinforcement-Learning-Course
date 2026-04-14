import numpy as np

from cartopole import CartPoleEnv
from DQN import DQNAgent
from tqdm import tqdm
from replay_buffer import ReplayBuffer


def train_dqn_naive(
    num_envs: int = 1,
    total_env_steps: int = 100_000,
    hidden_layers: list[int] | None = None,
    learning_rate: float = 1e-3,
    exploration_strategy: str = "epsilon_greedy",
    epsilon: float = 0.05,
    temperature: float = 1.0,
    gamma: float = 0.99,
    update_every: int = 1,
    seed: int = 42,
    device: str = "cpu",
    use_target_network: bool = False,
    target_update_every: int = 100,
    eval_every: int = 5000,
    n_eval_episodes: int = 10,
    use_replay_buffer: bool = False,
    replay_capacity: int = 10000,
    batch_size: int = 64,
    min_buffer_size: int = 1000,
):
    """
    Main DQN training loop for CartPole-v1.
    """
    env = CartPoleEnv(num_envs=num_envs, seed=seed)

    agent = DQNAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
        exploration_strategy=exploration_strategy,
        epsilon=epsilon,
        temperature=temperature,
        gamma=gamma,
        device=device,
        use_target_network=use_target_network,
        target_update_every=target_update_every,
    )

    if use_replay_buffer:
        replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            obs_dim=env.obs_dim,
        )
    else:
        replay_buffer = None

    obs, _ = env.reset()

    # Per-env running totals reset when an episode ends
    running_returns = np.zeros(num_envs, dtype=np.float32)
    completed_returns = []        # Episode returns logged when an episode finishes
    completed_return_steps = []   # Env step at which each episode finished
    steps_since_last_update = 0   # Tracks accumulated steps for the rollout update gate
    total_steps = 0
    losses = []
    evaluation_step_count = 0

    # Rollout accumulation buffers (only used when NOT using a replay buffer).
    # Each append adds one step's worth of data: shape (num_envs, ...).
    rollout_obs      = []
    rollout_actions  = []
    rollout_rewards  = []
    rollout_next_obs = []
    rollout_dones    = []

    pbar = tqdm(total=total_env_steps, desc="Training", unit="env_step")

    eval_returns = []
    eval_steps   = []
    next_log_step = 1000  # Step threshold for the next tqdm postfix update

    while total_steps < total_env_steps:

        # 1. Collect one step of experience from all envs
        actions = agent.select_actions(obs)
        next_obs, rewards, dones, infos = env.step(actions)
        train_next_obs = next_obs.copy()

        # For vectorized envs, a done env is auto-reset by Gymnasium. The true
        # terminal observation is stored in infos["final_observation"], not in
        # next_obs. We substitute it so the TD bootstrap uses the correct state.
        if num_envs > 1 and "final_observation" in infos:
            for i in range(num_envs):
                if dones[i] and infos["final_observation"][i] is not None:
                    train_next_obs[i] = infos["final_observation"][i]

        # 2. Store transitions
        if use_replay_buffer:
            # Add every transition from every env into the replay buffer
            for i in range(num_envs):
                replay_buffer.add(
                    obs=obs[i],
                    action=int(actions[i]),
                    reward=float(rewards[i]),
                    next_obs=train_next_obs[i],
                    done=bool(dones[i]),
                )
        else:
            # Accumulate this step's transitions for the rollout update
            rollout_obs.append(obs.copy())
            rollout_actions.append(actions.copy())
            rollout_rewards.append(rewards.copy())
            rollout_next_obs.append(train_next_obs.copy())
            rollout_dones.append(dones.copy())

        running_returns += rewards
        steps_since_last_update += num_envs

        # 3. Network update
        if use_replay_buffer:
            # Update every step once the buffer has enough data
            if len(replay_buffer) >= min_buffer_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.train_step(
                    obs=batch[0],
                    actions=batch[1],
                    rewards=batch[2],
                    next_obs=batch[3],
                    dones=batch[4],
                )
                losses.append(loss)
        else:
            # Update once enough steps have been accumulated in the rollout buffers
            if steps_since_last_update >= update_every:
                # Concatenate all rollout steps into one flat batch
                cat_obs      = np.concatenate(rollout_obs,      axis=0)
                cat_actions  = np.concatenate(rollout_actions,  axis=0)
                cat_rewards  = np.concatenate(rollout_rewards,  axis=0)
                cat_next_obs = np.concatenate(rollout_next_obs, axis=0)
                cat_dones    = np.concatenate(rollout_dones,    axis=0)

                loss = agent.train_step(
                    obs=cat_obs,
                    actions=cat_actions,
                    rewards=cat_rewards,
                    next_obs=cat_next_obs,
                    dones=cat_dones,
                )
                losses.append(loss)
                steps_since_last_update = 0

                # All accumulated transitions have been consumed — reset the buffers
                rollout_obs      = []
                rollout_actions  = []
                rollout_rewards  = []
                rollout_next_obs = []
                rollout_dones    = []

        # 4. Bookkeeping: episode returns, env reset, eval, logging
        for i in range(num_envs):
            if dones[i]:
                completed_returns.append(float(running_returns[i]))
                completed_return_steps.append(total_steps + num_envs)
                running_returns[i] = 0.0

        # For single-env mode Gymnasium does not auto-reset, so we do it manually
        if num_envs == 1 and dones[0]:
            next_obs, _ = env.reset()

        obs = next_obs
        total_steps += num_envs

        evaluation_step_count += num_envs

        if evaluation_step_count >= eval_every:
            mean_eval_return = agent.evaluate(n_episodes=n_eval_episodes, seed=seed + total_steps)
            eval_returns.append(mean_eval_return)
            eval_steps.append(total_steps)
            evaluation_step_count = 0

        # Compute recent statistics for the progress bar
        recent_mean_return = np.mean(completed_returns[-10:]) if completed_returns else 0.0
        recent_mean_loss   = np.mean(losses[-50:])            if losses            else 0.0

        if total_steps >= next_log_step:
            postfix = {
                "episodes": len(completed_returns),
                "ret10":    f"{recent_mean_return:.1f}",
                "loss50":   f"{recent_mean_loss:.4f}",
            }

            if eval_returns:
                postfix["eval"] = f"{eval_returns[-1]:.1f}"
            if use_replay_buffer:
                postfix["buffer"] = len(replay_buffer)
            if use_target_network:
                postfix["tn"] = "on"

            pbar.set_postfix(postfix)
            next_log_step += 1000

        pbar.update(num_envs)

    pbar.close()
    env.close()

    return completed_returns, completed_return_steps, losses, eval_returns, eval_steps
