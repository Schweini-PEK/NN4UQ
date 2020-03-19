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
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4
