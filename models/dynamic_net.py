import random

import torch
from torch import nn

from .basicmodule import BasicModule


class BasicFCBlock(BasicModule):
    def __init__(self, in_dim, out_dim):
        super(BasicFCBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())
        self.layer2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class DynamicNet(BasicModule):
    def __init__(self, in_dim, n_hidden, out_dim):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(in_dim, n_hidden)
        self.middle_linear = torch.nn.Linear(n_hidden, n_hidden)
        self.output_linear = torch.nn.Linear(n_hidden, out_dim)

    def forward(self, x):
        h = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h = self.middle_linear(h).clamp(min=0)
        return self.output_linear(h)


class DynamicMLP(BasicModule):
    def __init__(self, in_dim=3, n_hidden=3, h_dim=40, out_dim=1):
        super(DynamicMLP, self).__init__()
        self.in_dim = in_dim
        self.n_hidden = n_hidden
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.layer = nn.Sequential(self.make_layer())

    def make_layer(self, block=BasicFCBlock):
        layers = [block(self.in_dim, self.h_dim)]
        for _ in range(1, self.n_hidden):
            layers.append(block(self.h_dim, self.h_dim))
        return nn.Sequential(*layers, nn.Linear(self.h_dim, self.out_dim))

    def forward(self, x):
        return self.layer(x)
