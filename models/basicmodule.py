import time

import torch


class BasicModule(torch.nn.Module):
    """Implement load and save methods.

    """

    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 模型的默认名字

    def load(self, path):
        """Load a model from the path.

        :param path: The path of the model.
        :return: The model
        """
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """Save a model with a concatenation of the model's name and current time.

        :param name: The name of the model.
        :return: The name of the model that saved.
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name
