"""
Here are some kits.
"""

import csv
import os
import pickle
import random
import re

import numpy as np
import torch
from torch import nn

import models


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


def show_variable(func):
    config_list = []
    for k in vars(func):
        config_list.append('{}: {}'.format(k, vars(func).get(k)))

    return config_list


def load_trajectory(x_path, y_path, save_path):
    with open(x_path, 'r')as f_x:
        reader_x = csv.reader(f_x)
        data_x = list(reader_x)
        alphas, lengths = [], []
        k = 0
        for i in range(len(data_x)):
            if data_x[i][0] == '0':
                lengths.append(i - k)
                k = i
                alphas.append([float(j) for j in data_x[i]])
        lengths.append(len(data_x) - sum(lengths))
        lengths.remove(0)

    with open(y_path, 'r')as f_y:
        reader_y = csv.reader(f_y)
        # Get number of out_dim
        out_dim = len(next(reader_y))
        f_y.seek(0)

        alphas = np.array(alphas)[:, out_dim:].tolist()

        trajectories = []
        for i in range(len(lengths)):
            # Load each trajectory
            trajectory = []
            for j in range(lengths[i]):
                trajectory.append(next(reader_y))
            trajectories.append(np.array(trajectory).flatten().astype(np.float).tolist())

    data = (lengths, alphas, trajectories)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    return data


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


def get_pth_from_dir(root):
    models_path = [root + f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    return models_path


def get_pth_from_subdir(root):
    subs = [f.path for f in os.scandir(root) if f.is_dir()]
    models_path = []
    for sub in subs:
        files = os.listdir(sub)
        for file_name in files:
            if file_name.endswith('pth'):
                models_path.append(sub + '/' + file_name)

    return models_path


def columns2csv(output_path, *args):
    rows = zip(*args)
    rows2csv(output_path, rows)


def rows2csv(output_path, rows):
    with open(output_path, 'a+') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def load_model_from_path(path, in_dim, out_dim):
    """Load the PyTorch model based on the path (i.e., 'results/RSResNet_88_4_2.pth').

    :param in_dim: The input dimension of the model, should be equal to |variables| + |uncertainty parameters|
    :param out_dim: The output dimension of the model, should be equal to |variables|
    :param path: The path of the model.
    :return: The PyTorch model and its legend.
    """
    name = path.split('/')[-1].split('.')[0]
    module = name.split('_')[0]
    model, legend = None, None

    if module == 'RSResNet':
        nodes, layers, k = [int(i) for i in name.split('_')[1:]]
        model = getattr(models, module)(h_dim=nodes, n_h_layers=layers, k=k, block=models.BNResBlock,
                                        in_dim=in_dim, out_dim=out_dim)
        model.load(path)
        legend = '{}N{}L{}K'.format(nodes, layers, k)

    else:
        raise NameError('No such module: {}'.format(module))
    return model, legend


def get_io_dim(path):
    """Get the IO dimensions based on the path (i.e., 'dataset/truth_x3a5.pkl').

    The path must contain the following substring 'x1a1', where the numbers of variables and uncertainty parameters have
    been pointed out.
    :param path:
    :return: The input and output dimensions for the model.
    """
    path = path.split('/')[-1]
    out_dim = int(re.search('x(.*)a', path).group(1))
    in_dim = out_dim + int(re.search('(?<=a)(\d*)(?=\D)', path).group(1))
    return in_dim, out_dim
