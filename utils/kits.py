"""
Here are some kits I used to simplify the main program.
"""

import csv
import random

import numpy as np
import torch
from torch import nn


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def load_trajectory(x_path, y_path):
    with open(x_path, 'r')as x_all:
        x_reader = csv.reader(x_all)
        samples = list(x_reader)

    with open(y_path, 'r')as f_y:
        reader_y = csv.reader(f_y)

        for row in reader_y:
            print(row)
            break

        # Get number of out_dim
        out_dim = len(next(reader_y))
        f_y.seek(0)
        for row in reader_y:
            print(row)
            break

    with open(x_path, 'r')as x_all:
        alphas = set()
        starter = []
        count = 0
        x_reader = csv.reader(x_all)
        for row in x_reader:
            alpha = tuple(row[out_dim:])
            if alpha not in alpha:
                alphas.add(alpha)
        return alphas


def k_fold_index_gen(idx, k=5):
    """
        for fold in range(grid.k_fold):
        if grid.k_fold == 1:  # when k_fold is not just ONE FOLD.
            train_size = int(len(record) * grid.train_ratio)
            val_size = len(record) - train_size
            train_set, val_set = random_split(record, [train_size, val_size])
            data_train_loader = DataLoader(train_set, batch_size=grid.batch_size, num_workers=grid.num_workers)
            data_val_loader = DataLoader(val_set, batch_size=grid.batch_size, num_workers=grid.num_workers)
        else:
            indices = idx(range(len(record)))
            milestone = utils.kfold.k_fold_index_gen(indices, n=grid.k_fold)
            train_indices = indices[:milestone[fold]] + indices[milestone[fold + 1]:]
            val_indices = indices[milestone[fold]:milestone[fold + 1]]
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            data_train_loader = DataLoader(record, batch_size=grid.batch_size,
                                           num_workers=grid.num_workers, sampler=train_sampler)
            data_val_loader = DataLoader(record, batch_size=grid.batch_size,
                                         num_workers=grid.num_workers, sampler=val_sampler)
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
