import torch
from torch import nn

from utils import swish
from .basicmodule import BasicModule


class BasicResBlock(BasicModule):
    def __init__(self, in_dim, h_dim, out_dim, n_hidden_layer, activation='Swish'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
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

        out += identity[:self.out_dim] if len(identity.size()) == 1 else identity[:, :self.out_dim]

        return out


class NewResBlock(BasicResBlock):
    def forward(self, x):
        alpha = x[self.out_dim:] if len(x.size()) == 1 else x[:, self.out_dim:]
        out = super().forward(x)
        out = torch.cat((out, alpha), 0) if len(x.size()) == 1 else torch.cat((out, alpha), 1)
        return out


class BNResBlock(BasicResBlock):
    def __init__(self, in_dim, h_dim, out_dim, n_hidden_layer, activation='Swish'):
        super().__init__(in_dim, h_dim, out_dim, n_hidden_layer)
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


class ResNet(BasicModule):
    def __init__(self, in_dim, h_dim, out_dim, n_h_layers, block=BasicResBlock):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_h_layers = n_h_layers
        self.layer = self._make_layer(block)

    def _make_layer(self, block):
        layers = [block(self.in_dim, self.h_dim, self.out_dim, self.n_h_layers)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class RSResNet(BasicModule):
    def __init__(self, in_dim, h_dim, out_dim, k, n_h_layers, block=BasicResBlock):
        super().__init__()
        self.k = k
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_h_layers = n_h_layers
        self.layer = self._make_layer(block)

    def _make_layer(self, block):
        layers = [block(self.in_dim, self.h_dim, self.out_dim, self.n_h_layers)]
        in_dim = self.out_dim
        for _ in range(1, self.k):
            layers.append(block(in_dim, self.h_dim, self.out_dim, self.n_h_layers))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class RTResNet(BasicModule):
    def __init__(self, in_dim, h_dim, out_dim, k, n_h_layers, block=NewResBlock):
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
        for i in range(self.k):
            x = self.layer(x)

        return x[:self.out_dim] if len(x.size()) == 1 else x[:, :self.out_dim]


class NewRSResNet(RSResNet):
    def _make_layer(self, block=NewResBlock):
        layers = [NewResBlock(self.in_dim, self.h_dim, self.out_dim, self.n_h_layers)]
        for _ in range(1, self.k):
            layers.append(NewResBlock(self.in_dim, self.h_dim, self.out_dim, self.n_h_layers))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out[:self.out_dim] if len(x.size()) == 1 else out[:, :self.out_dim]
