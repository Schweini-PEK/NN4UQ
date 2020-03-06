import pickle

import numpy as np
from torch.utils.data import Dataset


class LoadDataset(Dataset):
    """A class to create dataset after pickle loading.

    """

    def __init__(self, test=False, root="dataset/", state='train', size=1000):
        self.state = state
        path = root + 'LSODE' + str(size) + '.pkl'
        if test:
            path = root
        with open(path, 'rb') as f:
            self.ode_frame = pickle.load(f)

    def __getitem__(self, item):
        x, y = self.ode_frame[item]
        x = np.array(x)
        y = np.array(y)
        if self.state == 'train':
            return x, y
        elif self.state == 'test':
            return x

    def __len__(self):
        return len(self.ode_frame)
