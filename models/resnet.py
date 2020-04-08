from torch import nn

from .basicmodule import BasicModule


class BasicBlock(BasicModule):
    def __init__(self, in_dim, n_hidden, out_dim=1, dropout=True):
        super(BasicBlock, self).__init__()
        self.dropout = dropout
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.Tanh())
        # self.layer4 = nn.Sequential(nn.Linear(n_hidden, out_dim), nn.Dropout(p=0.5))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden, out_dim))
        self.layer5 = nn.Dropout(p=0.5)

    def forward(self, x):
        identity = x
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        out = self.layer4(x3)
        if self.dropout:
            out = self.layer5(out)

        if len(identity.size()) == 1:
            out += identity[0]
        else:
            out += identity[:, 0].view(identity.size()[0], 1)

        return out


class ResNet(BasicModule):
    def __init__(self, block=BasicBlock, in_dim=3, n_hidden=30, k=1, out_dim=1):
        super(ResNet, self).__init__()
        self.k = k
        self.in_dim = in_dim
        self.n_hidden = n_hidden
        self.layer = self._make_layer(block, k, out_dim)

    def _make_layer(self, block, k, out_dim):
        layers = [block(self.in_dim, self.n_hidden, out_dim)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class RSResNet(ResNet):
    def __init__(self, in_dim=3, n_hidden=20, k=3, out_dim=1):
        super().__init__(BasicBlock, in_dim, n_hidden, k, out_dim)

    def _make_layer(self, block, k, out_dim):
        layers = [block(self.in_dim, self.n_hidden)]
        self.in_dim = out_dim
        for _ in range(1, k):
            layers.append(block(self.in_dim, self.n_hidden))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x
