import torch
from torch import nn


class Swish(nn.Module):
    """Custom the Swish activation function.

    """

    def __init__(self, beta=0.3):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(x * self.beta)
