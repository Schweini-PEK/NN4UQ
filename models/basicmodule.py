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

    def save(self, name=None, folder=None):
        """Save a model with a concatenation of the model's name and current time.

        :param folder: Saving to a specific folder.
        :param name: The name of the model.
        :return: The name of the model that saved.
        """
        base = 'checkpoints/' + folder
        if name is None:
            prefix = base + self.model_name + '_'
            name = time.strftime(prefix + '%H:%M.pth')
        torch.save(self.state_dict(), name)
        return name
