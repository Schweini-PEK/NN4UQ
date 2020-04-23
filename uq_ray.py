"""Checklist
1.The mode in
analysis.get_best_config(metric='val_loss', mode='min')
and the one in
scheduler=ASHAScheduler(metric='val_loss', mode='min',
                        max_t=1000, grace_period=200)
should be the same.

2. Both the max_t and the for loop decides the max of epochs.

3.


"""
import logging
import os

import torch
import torch.optim as optim
from hyperopt import hp
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

import models
import utils
from config import config
from uq_toy import train, val
from utils.data import loader

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(module)s - %(message)s')


def train_uq(grid):
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    path = os.path.dirname(os.path.realpath(__file__)) + '/dataset/reactor.pkl'
    data = utils.data.loader.LoadDataset(path=path)

    train_loader, test_loader = utils.data.loader.get_data_loaders(data,
                                                                   batch_size=int(grid['bs']),
                                                                   num_workers=4)

    in_dim, out_dim = len(data[0][0]), len(data[0][1])
    print('The input and output dimensions of models are {}, {}'.format(in_dim, out_dim))
    model = models.ResNet(in_dim=in_dim, out_dim=out_dim)
    optimizer = optim.SGD(
        model.parameters(),
        lr=grid["lr"])
    for i in range(1000):
        train(model, optimizer, train_loader, torch.device("cpu"))
        acc = val(model, test_loader, torch.device("cpu"))
        tune.track.log(val_loss=acc)
        if (i + 1) % 50 == 0:
            model.save(path="./model.pth")


if __name__ == '__main__':
    search_space = {
        "lr": tune.choice([0.01, 0.05, 0.1]),
        "bs": tune.choice([4, 8]),
    }

    space = {
        "lr": hp.uniform("lr", 0.01, 0.11),
        "bs": hp.randint("bs", 16, 128),
    }

    hyperopt_search = HyperOptSearch(
        space, max_concurrent=config.max_concurrent, metric="val_loss")

    analysis = tune.run(train_uq, config=search_space, num_samples=config.n_samples,
                        search_alg=hyperopt_search,
                        scheduler=ASHAScheduler(metric='val_loss', mode='min',
                                                max_t=1000, grace_period=50))
    dfs = analysis.trial_dataframes

    print("Best hyperparameters: ", analysis.get_best_config(metric='val_loss', mode='min'))

    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.val_loss.plot(ax=ax, legend=True)
