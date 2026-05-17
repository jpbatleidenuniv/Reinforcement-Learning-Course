import torch 

gamma = 0.9 

gammas = torch.pow(torch.tensor(gamma, dtype=torch.float32), torch.arange(10, dtype=torch.float32))
print(gammas)