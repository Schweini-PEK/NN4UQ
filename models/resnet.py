from torch import nn

from .basicmodule import BasicModule


class BasicBlock(BasicModule):
    def __init__(self, in_dim, n_hidden):
        super(BasicBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden), nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.Tanh())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.Tanh())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden, 2))

    def forward(self, x):
        identity = x

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out += identity

        return out


class ResNet(BasicModule):
    def __init__(self, block, inplanes, n_hidden, k, planes=1):
        super(ResNet, self).__init__()
        self.k = k
        self.inplanes = inplanes
        self.n_hidden = n_hidden
        self.layer = self._make_layer(block, planes, k)

    def _make_layer(self, block, planes, k):
        layers = [block(self.inplanes, self.n_hidden)]
        self.inplanes = planes
        for i in range(1, k):
            layers.append(block(self.inplanes, self.n_hidden))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)

        return x


def _resnet(block, inplanes, n_hidden, k, pretrained=False, arch=None, **kwargs):
    model = ResNet(block, inplanes, n_hidden, k)
    if pretrained:
        raise NotImplementedError("No idea")
    return model


def resnet(in_dim=2, n_hidden=30):
    model = _resnet(BasicBlock, in_dim, n_hidden, k=1)
    return model


def rs_resnet(in_dim=1, n_hidden=20, k=3):
    model = _resnet(BasicBlock, in_dim, n_hidden, k)
    return model

# def rt_resnet
