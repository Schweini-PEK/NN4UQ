import logging
import pickle

from torch.utils.data import Dataset, DataLoader, random_split


class LoadDataset(Dataset):
    def __init__(self, path, sample_freq=1):
        """

        :param path: The folder. "dataset/"
        :param sample_freq: The freq for sampling.
        """
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
            temp = []
            for i in range(self.__len__()):
                if i % sample_freq == 0:
                    temp.append(self.data[i])

            self.data = temp
            logging.info('{} dataset have been loaded.'.format(self.__len__()))

    def __getitem__(self, item):
        x, y = self.data[item]
        # x = np.array(x).astype(float)
        # y = np.array(y).astype(float)
        return x, y

    def __len__(self):
        return len(self.data)


def get_data_loaders(data, batch_size, ratio=0.8, num_workers=1):
    """Get dataset loaders for PyTorch models, which would be split to two parts according to ratio.

    :param data: Original dataset, should be a list.
    :param batch_size: The batch size.
    :param ratio: |training set| / |dataset|
    :param num_workers: The number of CPU cores to load dataset.
    :return: The training set loader and the validation set loader.
    """
    train_size = int(len(data) * ratio)
    val_size = len(data) - train_size
    train_set, val_set = random_split(data, [train_size, val_size])
    data_train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    data_val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return data_train_loader, data_val_loader
