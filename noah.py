import configparser
import logging
import os
import time

import fire
import ray
import torch
from hyperopt import hp
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import TerminateOnNan
from ignite.metrics import Loss
from ray import tune
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from torch import nn
from torch import optim

import models
import utils
from utils.data import loader
from utils.kits import setup_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SEED = 6


def ray_tuning(**kwargs):
    root = os.path.dirname(os.path.realpath(__file__)) + '/'
    config = configparser.ConfigParser()
    config.read(root + 'configure.ini')
    utils.kits.parse(config, 'tuning', kwargs)
    ray.init(num_cpus=24)
    if config.getboolean('tuning', 'sleep'):
        time.sleep(99)
    num_gpus = torch.cuda.device_count()
    metric = 'val_loss'

    def trial(grid):
        setup_seed(SEED)
        utils.kits.parse(config, 'tuning', grid)
        cfg = config['tuning']  # Current config

        n_nodes = int(float(cfg['n_nodes']))
        n_layers = int(float(cfg['n_layers']))
        k = int(float(cfg['k']))
        bs = int(float(cfg['bs']))

        data = utils.data.loader.LoadDataset(path=root + cfg['path'])
        train_loader, val_loader = utils.data.loader.get_data_loaders(data, batch_size=bs, num_workers=4)
        in_dim, out_dim = len(data[0][0]), len(data[0][1])

        if cfg['model'] in {'RSResNet', 'RTResNet', 'NewRSResNet'}:
            model = getattr(models, cfg['model'])(in_dim=in_dim, out_dim=out_dim,
                                                  k=k, n_h_layers=n_layers, h_dim=n_nodes)
            name = cfg['model'] + '_' + str(n_nodes) + '_' + str(n_layers) + '_' + str(k) + '.pth'

        elif cfg['model'] in {'ResNet', 'BNResNet'}:
            model = getattr(models, cfg['model'])(in_dim=in_dim, out_dim=out_dim, n_h_layers=n_layers, h_dim=n_nodes)
            name = cfg['model'] + '_' + str(n_nodes) + '_' + str(n_layers) + '.pth'

        else:
            raise NameError('No such model: {}'.format(cfg['model']))

        optimizer = getattr(optim, cfg['optimizer'])(model.parameters(), lr=float(cfg['lr']))
        criterion = nn.L1Loss()  # MAE Loss
        trainer = create_supervised_trainer(model, optimizer, criterion)
        val_metrics = {
            "L1": Loss(criterion)
        }
        evaluator = create_supervised_evaluator(model, metrics=val_metrics)
        print(model)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics
            model.save(path=name)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            tune.track.log(val_loss=metrics["L1"])

        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        trainer.run(train_loader, max_epochs=int(cfg['epoch']))
        model.save(path=name)

    if config['tuning']['scheduler'] == 'ASHA':
        scheduler = ASHAScheduler(metric=metric, mode='min', max_t=1000, grace_period=3)

    elif config['tuning']['scheduler'] == 'AHB':
        scheduler = AsyncHyperBandScheduler(metric=metric, mode="min", max_t=1000, grace_period=50)

    else:
        raise NameError('No such tuning scheduler: {}'.format(config['tuning']['scheduler']))

    if config['tuning']['strategy'] == 'BO':
        search_space = {'lr': (0.0001, 0.008), 'n_layers': (3, 8.99),
                        'n_nodes': (4, 30.99), 'k': (2, 4.99)}
        search_alg = BayesOptSearch(
            space=search_space, metric=metric, mode='min', max_concurrent=24,
            utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}
        )

    elif config['tuning']['strategy'] == 'hyperopt':
        search_space = {"lr": hp.uniform("lr", 0.001, 0.0099),
                        "n_layers": hp.randint("n_layers", 3, 9),
                        "n_nodes": hp.randint("n_nodes", 40, 200), "k": hp.randint("k", 2, 10)}
        search_alg = HyperOptSearch(search_space, metric=metric, max_concurrent=24, mode='min')

    else:
        raise NameError('No such optimizing strategy: {}'.format(config['tuning']['strategy']))

    analysis = tune.run(trial, name=config['tuning']['name'], search_alg=search_alg, scheduler=scheduler,
                        num_samples=config.getint('tuning', 'n_trials'),
                        resources_per_trial={"cpu": 2, "gpu": num_gpus / config.getint('tuning', 'concurrent')})
    dfs = analysis.trial_dataframes
    print("Best hyperparameter: {}".format(analysis.get_best_config(metric=metric, mode='min')))

    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.val_loss.plot(ax=ax, legend=True)


if __name__ == '__main__':
    fire.Fire(ray_tuning)
