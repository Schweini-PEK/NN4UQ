import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split


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


def print_best_score(gs, param_test):
    print("Best score: %0.3f" % gs.best_score_)
    print("Best parameters set:")
    best_parameters = gs.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def str2list(s: str):
    return np.array(s[1:-1].split(',')).astype(np.float)


def get_data_loaders(data, ratio, batch_size, num_workers):
    train_size = int(len(data) * ratio)
    val_size = len(data) - train_size
    train_set, val_set = random_split(data, [train_size, val_size])
    data_train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    data_val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    return data_train_loader, data_val_loader


def apply_dropout(m):
    if type(m) == nn.Dropout:
        print('FOUND')
        m.train()


def k_fold_index_gen(idx, k=5):
    """
        for fold in range(config.k_fold):
        if config.k_fold == 1:  # when k_fold is not just ONE FOLD.
            train_size = int(len(data) * config.train_ratio)
            val_size = len(data) - train_size
            train_set, val_set = random_split(data, [train_size, val_size])
            data_train_loader = DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers)
            data_val_loader = DataLoader(val_set, batch_size=config.batch_size, num_workers=config.num_workers)
        else:
            indices = idx(range(len(data)))
            milestone = utils.kfold.k_fold_index_gen(indices, n=config.k_fold)
            train_indices = indices[:milestone[fold]] + indices[milestone[fold + 1]:]
            val_indices = indices[milestone[fold]:milestone[fold + 1]]
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            data_train_loader = DataLoader(data, batch_size=config.batch_size,
                                           num_workers=config.num_workers, sampler=train_sampler)
            data_val_loader = DataLoader(data, batch_size=config.batch_size,
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
