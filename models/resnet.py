import torch
from torch import nn

from utils import swish
from .basicmodule import BasicModule


class BasicResBlock(BasicModule):
    def __init__(self, in_dim, h_dim, out_dim, n_hidden_layer, activation='Swish'):
        super(BasicResBlock, self).__init__()
        self.in_dim = in_dim
        self.activation = swish.Swish if activation == 'Swish' else getattr(nn, activation)
        layers_fc = [nn.Sequential(nn.Linear(in_dim, h_dim), self.activation())]
        for _ in range(1, n_hidden_layer):
            layers_fc.append(nn.Sequential(nn.Linear(h_dim, h_dim), self.activation()))
        self.layers_fc = nn.Sequential(*layers_fc)
        self.layer_dp = nn.Dropout(p=0.5)
        self.layer_out = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        identity = x
        x = self.layers_fc(x)
        x = self.layer_dp(x)
        out = self.layer_out(x)

        if len(identity.size()) == 1:
            out += identity[0]
        else:
            out += identity[:, 0].view(identity.size()[0], 1)
            # out += identity[:, :self.in_dim]

        return out


class BNResBlock(BasicModule):
    def __init__(self, in_dim, h_dim, out_dim, n_hidden_layer, activation='Swish'):
        super(BNResBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = swish.Swish if activation == 'Swish' else getattr(nn, activation)
        layers_fc = [nn.Sequential(nn.Linear(in_dim, h_dim), self.activation())]
        for _ in range(1, n_hidden_layer):
            layers_fc.append(nn.Sequential(nn.Linear(h_dim, h_dim),
                                           nn.BatchNorm1d(h_dim), self.activation()))
        self.layers_fc = nn.Sequential(*layers_fc)
        self.layer_dp = nn.Dropout(p=0.5)
        self.layer_out = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        identity = x
        x = self.layers_fc(x)
        x = self.layer_dp(x)
        out = self.layer_out(x)

        # len(out) = number of x

        if len(identity.size()) == 1:
            out += identity[:self.out_dim]
        else:
            out += identity[:, :self.out_dim].view(identity.size()[0], self.out_dim)

        return out


class ResNet(BasicModule):
    def __init__(self, block=BasicResBlock, in_dim=3, h_dim=30, out_dim=1, n_h_layers=3):
        super(ResNet, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_h_layers = n_h_layers
        self.layer = self._make_layer(block, out_dim, n_layer=n_h_layers)

    def _make_layer(self, block, out_dim, n_layer):
        layers = [block(self.in_dim, self.h_dim, out_dim, n_layer)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class RSResNet(BasicModule):
    def __init__(self, block=BasicResBlock, in_dim=6, h_dim=20, out_dim=1, k=3, n_h_layers=3):
        super(RSResNet, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_h_layers = n_h_layers
        self.layer = self._make_layer(block)

    def _make_layer(self, block):
        layers = [block(self.in_dim, self.h_dim, self.out_dim, self.n_h_layers)]
        self.in_dim = self.out_dim
        for _ in range(1, self.k):
            layers.append(block(self.in_dim, self.h_dim, self.out_dim, self.n_h_layers))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class RTResNet(BasicModule):
    def __init__(self, block=BasicResBlock, in_dim=3, h_dim=20, k=3, out_dim=1, n_h_layers=3):
        super().__init__()
        self.k = k
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_h_layers = n_h_layers
        self.layer = self._make_layer(block)

    def _make_layer(self, block):
        layers = [block(self.in_dim, self.h_dim, self.out_dim, self.n_h_layers)]
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


class BNRSResNet(BasicModule):
    def __init__(self, block=BasicResBlock, in_dim=6, h_dim=20, out_dim=1, k=3, n_h_layers=3):
        super(BNRSResNet, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_h_layers = n_h_layers
        self.layer = self._make_layer(block)

    def _make_layer(self, block):
        layers = [block(self.in_dim, self.h_dim, self.out_dim, self.n_h_layers)]
        self.in_dim = self.out_dim
        for _ in range(1, self.k):
            layers.append(block(self.in_dim, self.h_dim, self.out_dim, self.n_h_layers))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
