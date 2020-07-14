import collections
import logging
import time
import torch
import pickle
from sklearn.model_selection import ParameterGrid
from torch import optim
import models
import utils
import numpy as np
import math
import utils.analysis
from matplotlib import pyplot as plt
from matplotlib import colors
from config import config
from trainable import train, val, test
from utils.data import generator
from utils.data import loader

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


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
    # i.e., to use different dataset in a single experiments, simply add:
    # 'dp': ['dataset/you_name_it', ...]
    # The name has to be the same with those in grid.py
    # params = {'model': ['RSResNet'],  # The name of the model.
    params = {'model': ['RSResNet'],  # The name of the model.
              'lr': [0.1],
              'epoch': [10],
              'bs': [17],
              }  # Batch Size

    params = collections.OrderedDict(sorted(params.items()))
    grids = list(ParameterGrid(params))

    analyst = trainable(grids)


def forecast(save_fig=False):
    truth_path = config.truth_path
    in_dim, out_dim = utils.kits.get_io_dim(truth_path)
    model_path = 'results/RSResNet_88_4_2.pth'
    model, legend = utils.kits.load_model_from_path(model_path, in_dim, out_dim)
    try:
        f = open(truth_path, 'rb')

    except FileNotFoundError:
        raise FileNotFoundError('No such file: {}'.format(config.truth_path))

    trajectories = pickle.load(f)
    l_plot_grid = math.ceil(math.sqrt(len(trajectories)))
    w_plot_grid = math.ceil(len(trajectories) / l_plot_grid)

    c_map = plt.cm.get_cmap('gist_rainbow')
    c_norm = colors.Normalize(vmin=0, vmax=out_dim-1)
    scalar_map = plt.cm.ScalarMappable(norm=c_norm, cmap=c_map)

    for i, sample in enumerate(trajectories):
        alpha, trajectory = sample
        plt.subplot(l_plot_grid, w_plot_grid, i + 1)
        x0 = [0.0] * out_dim
        prediction = np.array(test(model, alpha, x0, len(trajectory) - 1))
        trajectory = np.array(trajectory)

        for j in range(out_dim):
            color = scalar_map.to_rgba(j)
            plt.plot(trajectory[:, j].transpose(), color=color)
            plt.plot(prediction[:, j].transpose(), color=color, marker='.', markersize=3)

    if save_fig:
        plt.savefig('test.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    forecast()
