import torch
from torch import nn

from utils import swish
from .basicmodule import BasicModule


class ShallowRes(BasicModule):
    """An implementation of the neural network, with batch normalization and ResNet.

    """

    def __init__(self, in_dim=3, n_hidden_1=40, n_hidden_2=40, out_dim=1):
        super(ShallowRes, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.Tanh())
        self.drop_layer1 = nn.Dropout(p=0.3)
        # The activation function of the 2nd layer is replaced by Swish.
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), swish.Swish())
        self.drop_layer2 = nn.Dropout(p=0.3)
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = self.drop_layer1(x1)
        x2 = self.layer2(x1)
        x2 = self.drop_layer1(x2)
        x3 = self.layer3(x2)
        return x3 + torch.unsqueeze(x[:, 0], 1)
