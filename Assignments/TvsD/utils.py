import torch


def argmax(
    obs: torch.Tensor, Q: torch.Tensor
) -> torch.Tensor:
    """Select the argmax Q_value of a given state and action"""
    try:
        return torch.random.choice(
            np.where(x == np.max(x))[0]
        )
    except:
        return torch.argmax
