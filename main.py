"""There are some things you have to do before getting through these scripts.
!pip install -r requirements.txt
!python -m visdom.server

To use tune train, run:
!python main.py tune_train

The configures in grid.py could be changed temporarily with commands in the following way:
!python main.py tune_train

To load data from .csv:
data_generator = generator.Generator()
data = data_generator.load_from_csv(x_path='dataset/X.csv', y_path='dataset/Y.csv', save=True, shuffle=False)

"""

import collections
import logging
import os
import pickle
import time

import numpy as np
import torch
import visdom
from sklearn.model_selection import ParameterGrid
from torch import optim

import models
import utils
import utils.experiment_analysis
from config import config
from uq_toy import train, val, test
from utils.data import generator
from utils.data import loader

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


def get_data():
    g = generator.Generator()
    g.load_from_csv(x_path='/Users/schweini/Downloads/y_1.csv',
                    name='tra_reactor')


def trainable(grids):
    # GPU
    device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
    logger.info('Using {}'.format(device))

    # Initialize analyst
    analyst = utils.experiment_analysis.Analysis(grids[0])
    init_time = time.time()
    result_folder = analyst.name

    # Go through every grid
    for grid in grids:
        config.parse(grid)  # Update grid into grid
        logger.info('Using grid: {}'.format(grid))

        data = utils.data.loader.LoadDataset(path=config.dp, sample_freq=config.sf)
        in_dim, out_dim = len(data[0][0]), len(data[0][1])
        logger.info('The input and output dimensions of models are {}, {}'.format(in_dim, out_dim))
        train_loader, test_loader = utils.data.loader.get_data_loaders(data, config.bs,
                                                                       config.ratio, num_workers=4)

        if config.model == 'RSResNet' or 'RTResNet':
            model = getattr(models, config.model)(in_dim=in_dim, out_dim=out_dim, k=config.k).to(device)
        else:
            # If you want to tune the number of nodes in hidden layers, here is an easier way to do so.
            model = getattr(models, config.model)(in_dim=in_dim, out_dim=out_dim).to(device)
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        optimizer = optim.SGD(model.parameters(), lr=config.lr)

        train_loss_list, val_loss_list = [], []
        t0 = time.time()
        for i in range(config.epoch):
            train_loss = train(model, optimizer, train_loader, device)
            train_loss_list.append(train_loss)

            if (i + 1) % config.val_freq == 0:
                val_loss = val(model, test_loader, device)
                val_loss_list.append(val_loss)

            if (i + 1) % config.print_freq == 0:
                logger.info('{i} epoch with {time}s'.format(i=i + 1, time=time.time() - t0))

        t = time.time() - t0
        checkpoint = model.save(folder=result_folder)
        record = {'train_loss': train_loss_list, 'val_loss': val_loss_list,
                  'ckpt': checkpoint, 'time': t}
        analyst.record(grid, record)

    logger.info('Experiments lasted {}s'.format(time.time() - init_time))
    analyst.save()
    return analyst


def tune_train(**kwargs):
    config.parse(kwargs)
    # Change the things here for the hyperparameter tuning.
    # Some other parts are also available to change.
    # i.e., to use different data in a single experiments, simply add:
    # 'dp': ['dataset/you_name_it', ...]
    # The name has to be the same with those in grid.py
    # params = {'model': ['RSResNet'],  # The name of the model.
    params = {'model': ['RTResNet'],  # The name of the model.
              'lr': [0.1],
              'epoch': [10],
              'bs': [17],
              }  # Batch Size

    params = collections.OrderedDict(sorted(params.items()))
    grids = list(ParameterGrid(params))

    analyst = trainable(grids)
    viz = visdom.Visdom()
    analyst.plot_loss(viz)
    analyst.predict(viz, config.state, test)


def plotter():
    """ This is a function to recall the results of the experiments.

    :return:
    """
    load_result_path = 'results/'
    analyst = utils.experiment_analysis.Analysis(pretrained=True, path=load_result_path)
    viz = visdom.Visdom()
    analyst.plot_loss(viz)
    # Change the dimensions of models if needed. This would be improved to be more automatically.
    # See more at the declaration.
    analyst.predict(viz, config.state, test)


def plot_from_ray():
    state = config.state
    viz = visdom.Visdom()
    length = state.get('length', 200)
    delta = state.get('delta', 0.1)
    win = '1'
    timeline = np.arange(0, length) * delta

    truth_path = 'dataset/tra_reactor.pkl'

    try:
        with open(truth_path, 'rb') as f:
            x_truth = np.array(pickle.load(f)).flatten().astype(np.float)
            viz.line(X=timeline, Y=x_truth,
                     win=win, name='truth',
                     opts=dict(title='reactor', legend=['truth'], showlegend=True))
    except FileNotFoundError:
        logging.warning('No ground truth available at {}.'.format(truth_path))

    root = '/Users/schweini/ray_results/question'
    # root = '/Users/schweini/ray_results/print'
    subs = [f.path for f in os.scandir(root) if f.is_dir()]
    for x in subs:
        path = x + '/model.pth'
        name = x[x.find('bs') + 3:x.find('bs') + 6] + '&' + x[x.find('lr') + 3:x.find('lr') + 11]
        model = models.ResNet(in_dim=6)
        model.load(path)
        test(model, state, viz, win=win, name=name, dropout=False)


if __name__ == '__main__':
    import fire

    fire.Fire()
