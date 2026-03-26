import torch
import numpy as np

from torch import nn
from torch import optim
from numpy.random import rand, randint, choice


class Loss(nn.Module): 
    def __init__(self,
                policy: str='epsilon-greedy',
                epsilon: float=0.1,
                temp: float=0.1,
                **kwargs) -> None:
        
        
        super().__init__(**kwargs)
        self.policy = policy
        self.epsilon = epsilon
        self.temp = temp
        assert(0 <= epsilon <= 1)
        

class NaiveLoss(Loss):
    def __init__(self, 
                 policy: str='epsilon-greedy',
                 epsilon: float=0.1,
                 temp: float=0.1,
                 **kwargs)-> None:
        
        super().__init__(policy, epsilon, temp, **kwargs)
                
    def loss(self, Q_sa: torch.Tensor, optimal_Q_sa_next: torch.Tensor, r: float):
        l = torch.square(r + optimal_Q_sa_next - Q_sa)
        return l
    


class NN(nn.Module):
    def __init__(self, 
                 hidden_layers: int,
                 width: int,
                 output_len: int=2,
                 input_len: int=4,
                 learning_rate: float=0.01,
                 **kwargs) -> None:
        
        super().__init__()
        self.h = hidden_layers
        self.w = width
        self.output_len = output_len
        self.input_len = input_len
        self.lr = learning_rate
        self.network = self._network()

        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.lr)

    def _network(self):
        network = nn.ModuleList()
        network.append(nn.Linear(self.input_len, self.w))
        network.append(nn.ReLU())

        for _ in range(self.h):
            network.append(nn.Linear(self.w, self.w))
            network.append(nn.ReLU())

        network.append(nn.Linear(self.w, self.output_len))
        network = nn.Sequential(*network)

        return network
    
    def forward(self, x: np.ndarray) -> torch.Tensor:
        x = self.network(torch.tensor(x))
        return x
    

class DQNAgent(NN, NaiveLoss):
    def __init__(self, 
                 hidden_layers: int, 
                 width: int, 
                 output_len: int=2, 
                 input_len: int=4, 
                 learning_rate: float=0.01, 
                 policy: str="epsilon-greedy", 
                 epsilon: float=0.01, 
                 temp: float=0.01,
                 gamma: float=0.99) -> None:

        super().__init__(
            hidden_layers=hidden_layers,
            width=width,
            output_len=output_len,
            input_len=input_len,
            learning_rate=learning_rate,
            policy=policy,
            epsilon=epsilon,
            temp=temp
        )

        self.gamma = gamma
        self._prev_Q_sa: torch.Tensor | None = None

        if self.policy not in ["epsilon-greedy", "softmax"]:
            raise ValueError("Policy must be either 'epsilon' or 'softmax'")


    @property
    def previous_Q_sa(self):
        return self._prev_optimal_Q_sa
    
    @previous_Q_sa.setter
    def previous_Q_sa(self, value: torch.Tensor):
        assert(isinstance(value, torch.Tensor))
        self._prev_optimal_Q_sa = value


    def action(self, Q_s: torch.Tensor, optimal: bool=False) -> tuple[int, torch.Tensor]:

        """Evaluates which action to choose following a policy, or the optimal policy"""

        optimal_Q_sa = torch.max(Q_s) * self.gamma # Needed for the loss

        if optimal:  # If we want follow the optimal policy we'll return the action that maximizes Q_s
            return int(torch.argmax(Q_s).item()), optimal_Q_sa

        # Otherwise return action following the policy
        a = self._policy(Q_s)
        assert(a is not None)

        return a, optimal_Q_sa
    
    
    def eval_Q(self, state: np.ndarray) -> torch.Tensor:
        Q_s = self.forward(state)
        return Q_s
    

    def _policy(self, Q_s) -> int | None:
        a = None

        if self.policy == "epsilon-greedy":
            if (rand() < self.epsilon):
                a = choice(range(self.output_len)).astype(int)
            else:
                a = int(torch.argmax(Q_s))

        elif self.policy == "softmax":
            probs = torch.softmax(Q_s / self.temp, dim=0).detach().numpy()
            a = int(choice(range(self.output_len), p=probs))

        return a
            
        
    



    

    