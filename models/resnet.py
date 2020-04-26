import torch
from torch import nn

from utils import swish
from .basicmodule import BasicModule


class BasicResBlock(BasicModule):
    def __init__(self, in_dim, n_hidden, out_dim=1, activation='Swish'):
        super(BasicResBlock, self).__init__()
        self.in_dim = in_dim
        self.activation = swish.Swish if activation == 'Swish' else getattr(nn, activation)
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden), self.activation())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden, n_hidden), self.activation())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden, n_hidden), self.activation())
        self.layer4 = nn.Dropout(p=0.5)
        self.layer5 = nn.Sequential(nn.Linear(n_hidden, out_dim))

    def forward(self, x):
        identity = x
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out = self.layer5(x4)

        if len(identity.size()) == 1:
            out += identity[0]
        else:
            out += identity[:, 0].view(identity.size()[0], 1)
            # out += identity[:, :self.in_dim]

        return out


class ResNet(BasicModule):
    def __init__(self, block=BasicResBlock, in_dim=3, n_hidden=30, out_dim=1):
        super(ResNet, self).__init__()
        self.in_dim = in_dim
        self.h_dim = n_hidden
        self.layer = self._make_layer(block, out_dim)

    def _make_layer(self, block, out_dim):
        layers = [block(self.in_dim, self.h_dim, out_dim)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class RSResNet(BasicModule):
    def __init__(self, block=BasicResBlock, in_dim=3, h_dim=20, k=3, out_dim=1):
        super(RSResNet, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.layer = self._make_layer(block)

    def _make_layer(self, block):
        layers = [block(self.in_dim, self.h_dim)]
        self.in_dim = self.out_dim
        for _ in range(1, self.k):
            layers.append(block(self.in_dim, self.h_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class RTResNet(BasicModule):
    def __init__(self, block=BasicResBlock, in_dim=3, h_dim=20, k=3, out_dim=1):
        super().__init__()
        self.k = k
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.layer = self._make_layer(block)

    def _make_layer(self, block):
        layers = [block(self.in_dim, self.h_dim)]
        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.size()) == 1:
            identity = x[self.in_dim:]
        else:
            identity = x[:, self.in_dim:]

        x = self.layer(x)
        for i in range(1, self.k):
            x = torch.cat((identity, x), 1)
            x = self.layer(x)
        return x
