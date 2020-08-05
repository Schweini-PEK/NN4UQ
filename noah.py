import configparser
import logging

import fire
import ray
import torch
import torch.optim as optim
from hyperopt import hp
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import TerminateOnNan
from ignite.metrics import Loss
from ray import tune
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from torch import nn

import models
import utils
from utils.data import loader
from utils.kits import setup_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SEED = 6


def ray_tuning(**kwargs):
    config = configparser.ConfigParser()
    config.read('configure.ini')
    utils.kits.parse(config, 'tuning', kwargs)
    ray.init(num_cpus=4)

    # time.sleep(99)
    num_gpus = torch.cuda.device_count()
    metric = 'val_loss'

    def trial(grid):
        setup_seed(SEED)
        utils.kits.parse(config, 'tuning', grid)
        cfg = config['tuning']  # Current config

        data = utils.data.loader.LoadDataset(path=cfg['path'])
        n_nodes = int(float(cfg['n_nodes']))
        n_layers = int(float(cfg['n_layers']))
        k = int(float(cfg['k']))
        bs = int(float(cfg['bs']))

        train_loader, val_loader = utils.data.loader.get_data_loaders(data, batch_size=bs, num_workers=4)

        in_dim, out_dim = len(data[0][0]), len(data[0][1])

        if cfg['model'] == 'RSResNet':
            model = models.RSResNet(in_dim=in_dim, out_dim=out_dim, k=k, n_h_layers=n_layers, h_dim=n_nodes)

            name = 'RSResNet_' + str(n_nodes) + '_' + str(n_layers) + '_' + str(k) + '.pth'

        elif cfg['model'] == 'RTResNet':
            model = models.RTResNet(in_dim=in_dim, out_dim=out_dim, k=k, n_h_layers=n_layers, h_dim=n_nodes)

            name = 'RTResNet_' + str(n_nodes) + '_' + str(n_layers) + '_' + str(k) + '.pth'

        elif cfg['model'] == 'NewRSResNet':
            model = models.NewRSResNet(in_dim=in_dim, out_dim=out_dim, k=k, n_h_layers=n_layers, h_dim=n_nodes,
                                       block=models.NewResBlock)

            name = 'NewRSResNet_' + str(n_nodes) + '_' + str(n_layers) + '_' + str(k) + '.pth'

        elif cfg['model'] == 'MLP':
            model = models.DynamicMLP(in_dim=in_dim, out_dim=out_dim, n_hidden=n_layers, h_dim=n_nodes)
            name = 'MLP_' + str(n_nodes) + '_' + str(n_layers) + '.pth'

        else:
            raise NameError('No such model: {}'.format(cfg['model']))

        optimizer = optim.SGD(
            model.parameters(),
            lr=grid["lr"])
        criterion = nn.L1Loss()  # MAE Loss
        trainer = create_supervised_trainer(model, optimizer, criterion)

        val_metrics = {
            "L1": Loss(criterion)
        }
        evaluator = create_supervised_evaluator(model, metrics=val_metrics)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            evaluator.run(train_loader)
            metrics = evaluator.state.metrics

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
        scheduler = AsyncHyperBandScheduler(metric="val_loss", mode="min", max_t=1000, grace_period=3)

    else:
        raise NameError('No such tuning scheduler: {}'.format(config['tuning']['scheduler']))

    if config['tuning']['strategy'] == 'BO':
        search_space = {'lr': (0.0001, 0.008), 'bs': (4, 64), 'n_layers': (3, 8.99),
                        'n_nodes': (4, 30.99), 'k': (2, 4.99)}
        search_alg = BayesOptSearch(
            space=search_space, metric='val_loss', mode='min', max_concurrent=20,
            utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}
        )

    elif config['tuning']['strategy'] == 'hyperopt':
        search_space = {"lr": hp.uniform("lr", 0.001, 0.0099), "bs": hp.randint("bs", 4, 48),
                        "n_layers": hp.randint("n_layers", 3, 9),
                        "n_nodes": hp.randint("n_nodes", 40, 200), "k": hp.randint("k", 2, 10)}
        search_alg = HyperOptSearch(search_space, metric="val_loss", max_concurrent=20, mode='min')

    else:
        raise NameError('No such optimizing strategy: {}'.format(config['tuning']['strategy']))

    analysis = tune.run(trial, name=config['tuning']['name'], search_alg=search_alg, scheduler=scheduler,
                        num_samples=config.getint('tuning', 'n_trials'),
                        resources_per_trial={"cpu": 1, "gpu": num_gpus / config.getint('tuning', 'concurrent')})
    dfs = analysis.trial_dataframes
    logger.info("Best hyperparameter: {}".format(analysis.get_best_config(metric='val_loss', mode='min')))

    # Plot by epoch
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.val_loss.plot(ax=ax, legend=True)


if __name__ == '__main__':
    fire.Fire(ray_tuning)
