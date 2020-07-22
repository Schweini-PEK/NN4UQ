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

import ray
import torch
import torch.optim as optim
from hyperopt import hp
from ray import tune
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

import models
import utils
from config import config
from config import config as cfg
from trainable import train, val
from utils.data import loader
from utils.kits import setup_seed

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(module)s - %(message)s')
SEED = 6


def train_uq(grid):
    setup_seed(SEED)
    cfg.parse(grid)

    data = utils.data.loader.LoadDataset(path=config.data_path)
    nodes = int(grid['nodes'])
    layers = int(grid['layers'])
    k = int(grid['k'])
    bs = int(grid['bs'])

    train_loader, test_loader = utils.data.loader.get_data_loaders(data,
                                                                   batch_size=bs,
                                                                   num_workers=4)

    # in_dim, out_dim = len(data[0][0]), len(data[0][1])
    in_dim, out_dim = 8, 3

    model = None

    if config.model == 'RSResNet':
        model = models.RSResNet(in_dim=in_dim, out_dim=out_dim, k=k,
                                n_h_layers=layers, h_dim=nodes, block=models.BNResBlock)

        name = 'RSResNet_' + str(nodes) + '_' + str(layers) + '_' + str(k) + '.pth'

    elif config.model == 'RTResNet':
        model = models.RTResNet(in_dim=in_dim, out_dim=out_dim, k=k,
                                n_h_layers=layers, h_dim=nodes, block=models.BNResBlock)

        name = 'RTResNet_' + str(nodes) + '_' + str(layers) + '_' + str(k) + '.pth'

    else:
        raise NameError('No such model: {}'.format(config.model))

    optimizer = optim.SGD(
        model.parameters(),
        lr=grid["lr"])
    # for i in range(int(grid['epoch'])):
    for i in range(400):
        train(model, optimizer, train_loader, torch.device("cpu"))
        acc = val(model, test_loader, torch.device("cpu"))
        tune.track.log(val_loss=acc)
        if (i + 1) % 5 == 0:
            model.save(path=name)
    model.save(path=name)


if __name__ == '__main__':
    ray.init(num_cpus=10)
    # time.sleep(60)
    num_gpus = torch.cuda.device_count()

    # Ray Tune Grid Search
    search_space = {
        "lr": tune.choice([0.01, 0.05, 0.1]),
        "bs": tune.choice([4, 8]),
    }

    # HyperOpt
    hyperopt_space = {
        "lr": hp.uniform("lr", 0.001, 0.0099),
        "bs": hp.randint("bs", 4, 48),
        # "epoch": hp.randint("epoch", 30, 70),
        "layers": hp.randint("layers", 3, 9),
        "nodes": hp.randint("nodes", 40, 200),
        "k": hp.randint("k", 2, 10)
    }
    hyperopt_search_alg = HyperOptSearch(
        hyperopt_space, metric="val_loss", mode='min')
    hyperopt_search_alg = tune.suggest.ConcurrencyLimiter(hyperopt_search_alg, max_concurrent=20)

    # Bayesian
    bo_space = {'lr': (0.0001, 0.008), 'bs': (4, 64), 'epoch': (20, 30.99), 'layers': (3, 8.99), 'nodes': (4, 30.99),
                'k': (2, 4.99)}
    bo_search_alg = BayesOptSearch(
        space=bo_space,
        metric='val_loss',
        mode='min',
        utility_kwargs={
            "kind": "ucb",
            "kappa": 2.5,
            "xi": 0.0
        }
    )
    bo_search_alg = tune.suggest.ConcurrencyLimiter(bo_search_alg, max_concurrent=20)

    ahb_scheduler = AsyncHyperBandScheduler(metric="val_loss", mode="min")
    ashas_scheduler = ASHAScheduler(metric='val_loss', mode='min', max_t=1000, grace_period=6)

    analysis = tune.run(train_uq, name='delete', search_alg=hyperopt_search_alg, scheduler=ashas_scheduler,
                        num_samples=1, resources_per_trial={"cpu": 1, "gpu": num_gpus})
    dfs = analysis.trial_dataframes

    print("Best hyperparameters: ", analysis.get_best_config(metric='val_loss', mode='min'))

    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.val_loss.plot(ax=ax, legend=True)
