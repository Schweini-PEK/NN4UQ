"""
Here are some kits I used to simplify the main program.
"""

import numpy as np
from torch import nn


def list2sample(n_features, dataset):
    """Get x and y(should be one value for one sample) in the numpy array format from the dataset.

    :param n_features: Number of features
    :param dataset: The original dataset
    :return: array-like of shape (n_samples, n_features) and array-like of shape (n_samples, 1)
    """
    n_samples = len(dataset)
    sample_x = np.zeros((n_samples, n_features))
    sample_y = np.zeros((n_samples, 1))

    for i in range(n_samples):
        for j in range(n_features):
            sample_x[i][j] = dataset[i][0][j]
        sample_y[i][0] = dataset[i][1]

    return sample_x, sample_y


def str2list(s: str):
    return np.array(s[1:-1].split(',')).astype(np.float)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def print_variable(func):
    config_list = []
    for k in vars(func):
        # if not k.startswith('__'):
        #     config_list.append('{}: {}'.format(k, vars(func).get(k)))
        config_list.append('{}: {}'.format(k, vars(func).get(k)))

    return config_list


def k_fold_index_gen(idx, k=5):
    """
        for fold in range(config.k_fold):
        if config.k_fold == 1:  # when k_fold is not just ONE FOLD.
            train_size = int(len(record) * config.train_ratio)
            val_size = len(record) - train_size
            train_set, val_set = random_split(record, [train_size, val_size])
            data_train_loader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers)
            data_val_loader = DataLoader(val_set, batch_size=config.batch_size, num_workers=config.num_workers)
        else:
            indices = idx(range(len(record)))
            milestone = utils.kfold.k_fold_index_gen(indices, n=config.k_fold)
            train_indices = indices[:milestone[fold]] + indices[milestone[fold + 1]:]
            val_indices = indices[milestone[fold]:milestone[fold + 1]]
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            data_train_loader = DataLoader(record, batch_size=config.batch_size,
                                           num_workers=config.num_workers, sampler=train_sampler)
            data_val_loader = DataLoader(record, batch_size=config.batch_size,
                                         num_workers=config.num_workers, sampler=val_sampler)
    :param idx:
    :param k:
    :return:
    """
    split = len(idx) // k
    milestone = [0]
    for i in range(1, k):
        milestone.append(i * split)
    milestone.append(len(idx))
    return milestone
