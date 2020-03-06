from torch import nn


class Swish(nn.Module):
    """Custom the Swish activation function.

    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * nn.functional.sigmoid(x)
