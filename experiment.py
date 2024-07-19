import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.mlp(x)


if __name__ == "__main__":

    # a = torch.randn((1668, 64))
    # model = MLP(a.shape[0] * a.shape[1], 3000)
    # a = a.view(a.shape[0]*a.shape[1])
    # print(a.shape)
    # x = model(a)
    # print(x.shape)
    a = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    print(a)
    a += 2
    print(a)