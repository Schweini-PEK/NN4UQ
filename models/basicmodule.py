import time

import torch


class BasicModule(torch.nn.Module):
    """Implement load and save methods.

    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self)).split("'")[1].split(".")[-1]

    def load(self, path):
        """Load a model from the path.

        :param path: The path of the model.
        :return: The model
        """
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        """Save a model by its state dict.

        :param path:
        :return: The name of the model.
        """
        torch.save(self.state_dict(), path)
        return path
