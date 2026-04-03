from gymnasium import Env
import torch
import numpy as np
import copy

from torch import nn
from torch import optim
from numpy.random import rand, choice


class ExperienceReplay():
    def __init__(self,
                 buffer: bool = False,
                 buffer_size: int = 100000,
                 min_buffer_size: int = 1000,
                 batch_size: int = 200,
                 device: str = "cpu") -> None:
        
        self.use_buffer = buffer
        self.max_buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.buffer = {}
        self.total_steps_seen = 0
        self.len_buffer = 0
        self.batch_size = batch_size
        self.device = device

    def get_sequence(self, 
                     state: np.ndarray, 
                     next_state: np.ndarray, 
                     action: int,
                     reward: int|float, 
                     done: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        
        # Create the transition tuple
        transition = (state, action, next_state, reward, done)

        if not self.use_buffer:
            return (
                torch.tensor(np.array([state]), dtype=torch.float32).to(device=self.device),
                torch.tensor([action], dtype=torch.long).to(device=self.device),
                torch.tensor(np.array([next_state]), dtype=torch.float32).to(device=self.device),
                torch.tensor([reward], dtype=torch.float32).to(device=self.device),
                torch.tensor([done], dtype=torch.bool).to(device=self.device)
            )

        # Use modulo to wrap the index back to 0 when it hits max_buffer_size, this will always replace the oldest transition tuple in the buffer
        write_index = self.total_steps_seen % self.max_buffer_size
        self.buffer[write_index] = transition
        self.total_steps_seen += 1

        # If the buffer is smaller than the minimal buffer size, we dont train
        if self.total_steps_seen < self.min_buffer_size:
            return None
        
        self.len_buffer = min(self.total_steps_seen, self.max_buffer_size)

        # Sample random transitions
        idx = np.random.randint(0, self.len_buffer, self.batch_size)
        batch = [self.buffer[i] for i in idx]

        # Makes tuples of lists 
        states, actions, next_states, rewards, dones = zip(*batch)

        tensor_batch = (torch.tensor(np.array(states), dtype=torch.float32).to(device=self.device),
                        torch.tensor(actions, dtype=torch.long).to(device=self.device),
                        torch.tensor(np.array(next_states), dtype=torch.float32).to(device=self.device),
                        torch.tensor(np.array(rewards), dtype=torch.float32).to(device=self.device),
                        torch.tensor(dones, dtype=torch.bool).to(device=self.device))
        
        return tensor_batch



class NN(nn.Module):
    def __init__(
        self,
        hidden_layers: int,
        width: int,
        learning_rate: float = 0.01,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.h = hidden_layers
        self.w = width
        self.lr = learning_rate
        self.network = self._build_network()
        self.device = device

        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=self.lr
        )

    def _build_network(self) -> nn.Sequential:
        layers = [
            nn.Linear(4, self.w),
            nn.ReLU(),
        ]
        for _ in range(self.h):
            layers += [nn.Linear(self.w, self.w), nn.ReLU()]
        layers.append(nn.Linear(self.w, 2))
        return nn.Sequential(*layers)

    def forward(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)
        return self.network(x)


class DQNAgent(NN):
    def __init__(
        self,
        hidden_layers: int,
        width: int,
        learning_rate: float = 0.01,
        policy: str = "epsilon-greedy",
        epsilon: float = 0.01,
        temp: float = 0.01,
        gamma: float = 0.99,
        target: bool = False,
        update_frequence: int = 100,
        device: str = "cpu",
        loss_function: str = "MSE"
    ) -> None:
        
        super().__init__(
            hidden_layers=hidden_layers,
            width=width,
            learning_rate=learning_rate,
            device=device
        )

        self.temp = temp
        self.policy = policy
        self.epsilon = epsilon
        self.update_frequence = update_frequence
        self.gamma = gamma
        self.target = target

        losses = {"MSE": nn.MSELoss, "MAE": nn.L1Loss}
        assert self.loss_function in losses, "Loss function can be MSE or MAE"
        self.loss_function = losses[loss_function]


        if self.target:
            self.target_network = copy.deepcopy(self.network)

        if self.policy not in ["epsilon-greedy", "softmax"]:
            raise ValueError(
                "Policy must be 'epsilon-greedy' or 'softmax'"
            )


    def action(
        self, Q_s: torch.Tensor, optimal: bool = False
    ) -> tuple[int, torch.Tensor]:
        """Choose an action following the current policy (or the greedy policy)."""

        optimal_Q_sa = torch.max(Q_s)
        a = self._policy(Q_s)

        if optimal:
            a = int(torch.argmax(Q_s).item())

        assert a is not None
        return a, optimal_Q_sa


    def eval_Q(self, state: torch.Tensor, target=False) -> torch.Tensor:
        if target:
            return self.target_network(state)

        return self.forward(state)


    def _policy(self, Q_s: torch.Tensor) -> int | None:
        if self.policy == "epsilon-greedy":
            if rand() < self.epsilon:
                return int(choice(range(2)))
            return int(torch.argmax(Q_s).item())

        if self.policy == "softmax":
            probs = (
                torch.softmax(Q_s / self.temp, dim=0)
                .detach()
                .numpy()
            )
            return int(choice(range(2), p=probs))

        return None

    def evaluate(
        self, eval_env: Env, num_episodes: int = 100
    ) -> float:
        returns = []
        for _ in range(num_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            episode_over = False

            while not episode_over:
                with torch.no_grad():
                    Q_s = self.eval_Q(state=obs)
                    action, _ = self.action(Q_s, optimal=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    
                episode_reward += float(reward)
                episode_over = terminated or truncated

            returns.append(episode_reward)

        return float(np.mean(returns))

    def loss(
        self,
        sequences: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
        count: int
        ) -> torch.Tensor | None:
        
        if sequences is None:
            return None

        states, actions, next_states, rewards, dones = sequences

        # Compute Q(s,a) using the current network
        Q_s = self.eval_Q(states)  # eval_Q expects numpy array
        Q_sa = Q_s.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():  # target should be a fixed value, not part of the graph
            Q_next = self.eval_Q(next_states, target=self.target) # Shape: (200, 2)
            max_Q_next = torch.max(Q_next, dim=1)[0]

            targets = rewards + self.gamma * max_Q_next * (~dones).float()

        l = self.loss_function(Q_sa, targets)

        # Handle target network updates outside any loop
        if self.target and (count % self.update_frequence == 0):
            self.target_network = copy.deepcopy(self.network)

        return l
