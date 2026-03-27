import torch
import numpy as np
import copy

from torch import nn
from torch import optim
from numpy.random import rand, choice


class Loss(nn.Module):
    def __init__(self,
                 policy: str = 'epsilon-greedy',
                 epsilon: float = 0.1,
                 temp: float = 0.1,
                 **kwargs) -> None:

        super().__init__(**kwargs)   # passes any remaining kwargs to nn.Module
        self.policy = policy
        self.epsilon = epsilon
        self.temp = temp
        assert 0 <= epsilon <= 1


class NaiveLoss(Loss):
    def __init__(self,
                 policy: str = 'epsilon-greedy',
                 epsilon: float = 0.1,
                 temp: float = 0.1,
                 **kwargs) -> None:

        super().__init__(policy=policy, epsilon=epsilon, temp=temp, **kwargs)

    def loss_naive(self, Q_sa: torch.Tensor, optimal_Q_sa_next: torch.Tensor,
                   r: torch.Tensor, gamma: float) -> torch.Tensor:
        # FIX 2: gamma applied here in the Bellman target, not inside action()
        l = torch.square(r + gamma * optimal_Q_sa_next - Q_sa)
        return l
        
class TargetNetworkLoss(Loss):
    def __init__(self,
                 policy: str = 'epsilon-greedy',
                 epsilon: float = 0.1,
                 temp: float = 0.1,
                 update_count: int = 1,
                 **kwargs) -> None:

        super().__init__(policy=policy, epsilon=epsilon, temp=temp, **kwargs)
        self.update_count = update_count

    def loss_target(self, Q_sa: torch.Tensor, optimal_Q_sa_next: torch.Tensor,
                    r: torch.Tensor, gamma: float, count: int) -> tuple[torch.Tensor, bool]:
        # FIX 2: gamma applied here in the Bellman target, not inside action()
        l = torch.square(r + gamma * optimal_Q_sa_next - Q_sa)
        update = (count % self.update_count == 0)
        return l, update
    

class Buffer():
    def __init__(self, buffer_size: int = 200) -> None:
        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_len = 0
        self.position = 0

    def pop(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Store an experience. If buffer is full, replace a random slot and return the evicted experience.
        All tensors are detached and stored as tensors.
        """
        # Convert to tensors (detach not needed because state/action are not from torch)
        state_t = torch.tensor(state, dtype=torch.float32)
        next_state_t = torch.tensor(next_state, dtype=torch.float32)
        action_t = torch.tensor(action, dtype=torch.long)
        reward_t = torch.tensor(reward, dtype=torch.float32)
        done_t = torch.tensor(done, dtype=torch.bool)

        if self.buffer_len < self.buffer_size:
            self.buffer.append((state_t, action_t, reward_t, next_state_t, done_t))
            self.buffer_len += 1
            return None
        else:
            idx = np.random.randint(0, self.buffer_size)
            evicted = self.buffer[idx]
            self.buffer[idx] = (state_t, action_t, reward_t, next_state_t, done_t)
            return evicted

    def clear(self):
        """Remove and return a random experience, or None if empty."""
        if self.buffer_len == 0:
            return None
        idx = np.random.randint(0, self.buffer_len)
        evicted = self.buffer.pop(idx)
        self.buffer_len -= 1
        return evicted



class NN(nn.Module):
    def __init__(self,
                 hidden_layers: int,
                 width: int,
                 output_len: int = 2,
                 input_len: int = 4,
                 learning_rate: float = 0.01,
                 **kwargs) -> None:


        super().__init__(**kwargs)

        self.h = hidden_layers
        self.w = width
        self.output_len = output_len
        self.input_len = input_len
        self.lr = learning_rate
        self.network = self._build_network()
        self._target_network = None

        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.lr)

    def update_target(self):
        self._target_network = copy.deepcopy(self.network)

    def _build_network(self) -> nn.Sequential:
        layers = [nn.Linear(self.input_len, self.w), nn.ReLU()]
        for _ in range(self.h):
            layers += [nn.Linear(self.w, self.w), nn.ReLU()]
        layers.append(nn.Linear(self.w, self.output_len))
        return nn.Sequential(*layers)

    def forward(self, x: np.ndarray, target: bool = False) -> torch.Tensor:
        x_tensor = torch.tensor(x, dtype=torch.float32)
        if target:
            assert self._target_network is not None
            with torch.no_grad():
                return self._target_network(x_tensor)
        return self.network(x_tensor)


class DQNAgent(NN, NaiveLoss, TargetNetworkLoss):
    def __init__(self,
                 hidden_layers: int,
                 width: int,
                 output_len: int = 2,
                 input_len: int = 4,
                 learning_rate: float = 0.01,
                 policy: str = "epsilon-greedy",
                 epsilon: float = 0.01,
                 temp: float = 0.01,
                 gamma: float = 0.99,
                 target_network: bool = False,
                 update_count: int = 1,
                 buffer: bool = False,
                 buffer_size: int = 200) -> None:
        
        super().__init__(
            hidden_layers=hidden_layers,
            width=width,
            output_len=output_len,
            input_len=input_len,
            learning_rate=learning_rate,
            policy=policy,
            epsilon=epsilon,
            temp=temp,
            update_count=update_count,
        )

        self.use_buffer = buffer
        self.keep_target_network = target_network
        self.gamma = gamma

        if buffer:
            self.buffer = Buffer(buffer_size=buffer_size)

        if target_network:
            self.update_target()  # initialise target network weights

        if self.policy not in ["epsilon-greedy", "softmax"]:
            raise ValueError("Policy must be 'epsilon-greedy' or 'softmax'")

    @property
    def previous_Q_sa(self):
        return self._prev_optimal_Q_sa

    @previous_Q_sa.setter
    def previous_Q_sa(self, value: torch.Tensor):
        assert isinstance(value, torch.Tensor)
        self._prev_optimal_Q_sa = value

    def action(self, Q_s: torch.Tensor, optimal: bool = False) -> tuple[int, torch.Tensor]:
        """Choose an action following the current policy (or the greedy policy)."""

        optimal_Q_sa = torch.max(Q_s)

        if optimal:
            return int(torch.argmax(Q_s).item()), optimal_Q_sa

        a = self._policy(Q_s)
        assert a is not None
        return a, optimal_Q_sa

    def eval_Q(self, state: np.ndarray, target: bool = False) -> torch.Tensor:
        return self.forward(state, target)

    def _policy(self, Q_s: torch.Tensor) -> int | None:
        if self.policy == "epsilon-greedy":
            if rand() < self.epsilon:
                return int(choice(range(self.output_len)))
            return int(torch.argmax(Q_s).item())

        if self.policy == "softmax":
            probs = torch.softmax(Q_s / self.temp, dim=0).detach().numpy()
            return int(choice(range(self.output_len), p=probs))

        return None

    def loss(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool,
            count: int | None = None) -> torch.Tensor | None:

        if self.use_buffer:
            result = self.buffer.pop(state, action, reward, next_state, done)
            if result is None:
                return None
            state_t, action_t, reward_t, next_state_t, done_t = result
        else:
            # If buffer is not used, convert inputs to tensors
            state_t = torch.tensor(state, dtype=torch.float32)
            action_t = torch.tensor(action, dtype=torch.long)
            reward_t = torch.tensor(reward, dtype=torch.float32)
            next_state_t = torch.tensor(next_state, dtype=torch.float32)
            done_t = torch.tensor(done, dtype=torch.bool)

        # Compute Q(s,a) using the current network
        Q_s = self.eval_Q(state_t.numpy())  # eval_Q expects numpy array
        Q_sa = Q_s[action_t.item()]

        # Compute optimal Q(s',a') using target network if requested, else current network
        if self.keep_target_network:
            # Use target network for the next state
            Q_s_next = self.eval_Q(next_state_t.numpy(), target=True)
        else:
            Q_s_next = self.eval_Q(next_state_t.numpy(), target=False)

        optimal_Q_sa_next = torch.max(Q_s_next) if not done_t.item() else torch.tensor(0.0, dtype=torch.float32)

        # Compute loss
        if self.keep_target_network:
            assert count is not None, "count is required when using a target network"
            l, update = self.loss_target(
                Q_sa=Q_sa, optimal_Q_sa_next=optimal_Q_sa_next,
                r=reward_t, gamma=self.gamma, count=count
            )
            if update:
                self.update_target()
        else:
            l = self.loss_naive(
                Q_sa=Q_sa, optimal_Q_sa_next=optimal_Q_sa_next,
                r=reward_t, gamma=self.gamma
            )

        return l