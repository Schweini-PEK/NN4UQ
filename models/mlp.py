from torch import nn

from utils import swish
from .basicmodule import BasicModule


class MLP(BasicModule):
    """A simple MLP, with only fc layers.

    """

    def __init__(self, in_dim=3, n_hidden_1=40, n_hidden_2=40, n_hidden_3=40, out_dim=1):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), swish.Swish())
        # The activation function of the 2nd layer is replaced by Swish.
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), swish.Swish())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), swish.Swish())
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, out_dim))

    def forward(self, x):
        x1 = self.layer1(x)
        print("x", x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4


class ShallowResBN(BasicModule):
    """An implementation of the neural network, with batch normalization and ResNet.

    """

    def __init__(self, in_dim=3, n_hidden_1=40, n_hidden_2=40, out_dim=1):
        super(ShallowResBN, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.swish1 = swish.Swish()
        self.drop_layer1 = nn.Dropout(p=0.3)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)
        self.swish2 = swish.Swish()
        self.drop_layer2 = nn.Dropout(p=0.3)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        if len(x.size()) == 1:
            x = x.unsqueeze(0)
        print("x", x)
        x1 = self.layer1(x)
        print("x1", x1)

        x1 = self.bn1(x1)
        x1 = self.swish1(x1)
        x1 = self.drop_layer1(x1)
        x2 = self.layer2(x1)
        x2 = self.bn2(x2)
        x2 = self.swish2(x2)
        x2 = self.drop_layer2(x2)
        x3 = self.layer3(x2)
        return x3
