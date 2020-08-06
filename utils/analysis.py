import json
import logging
import math
import pickle
import re
import time

import numpy as np
from matplotlib import pyplot as plt

import models
from trainable import test
from utils import kits

logger = logging.getLogger(__name__)


def get_loss_from_ray(path):
    """Read and return a list of loss from 'RT_400.json' generated by Ray.Tune

    :param path: The pat of 'RT_400.json'
    :return: A list of loss
    """

    losses = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            losses.append(line_dict['val_loss'])

    return losses


def load_model_from_path(path, in_dim, out_dim):
    """Load the PyTorch model based on the path (i.e., 'results/RSResNet_88_4_2.pth').

    :param in_dim: The input dimension of the model, should be equal to |variables| + |uncertainty parameters|
    :param out_dim: The output dimension of the model, should be equal to |variables|
    :param path: The path of the model.
    :return: The PyTorch model and its legend.
    """
    name = path.split('/')[-1].split('.')[0]
    module = name.split('_')[0]

    if module in {'NewRSResNet', 'RTResNet', 'RSResNet'}:
        nodes, layers, k = [int(i) for i in name.split('_')[1:]]
        model = getattr(models, module)(h_dim=nodes, n_h_layers=layers, k=k, in_dim=in_dim, out_dim=out_dim)
        caption = '{}{}N{}L{}K'.format(module, nodes, layers, k)

    else:
        raise NameError('No such module: {}'.format(module))

    model.load(path)
    return model, caption


def forecast(model_path, truth_path, save_fig=False):
    # torch.manual_seed(6)

    in_dim, out_dim = get_io_dim(truth_path)
    model, name = load_model_from_path(model_path, in_dim, out_dim)

    try:
        f = open(truth_path, 'rb')
    except FileNotFoundError:
        raise ('No such file: {}'.format(truth_path))

    trajectories = pickle.load(f)
    l_plot_grid = math.ceil(math.sqrt(len(trajectories)))
    w_plot_grid = math.ceil(len(trajectories) / l_plot_grid)
    scalar_map = kits.color_wheel(out_dim)
    fig, axes = plt.subplots(l_plot_grid, w_plot_grid, squeeze=False, sharex=True, sharey=True)
    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Quantities of Interest', va='center', rotation='vertical')
    fig.suptitle('Predictions with model {}'.format(name), y=0.02)

    e_t, t_t = [], []
    for i, sample in enumerate(trajectories):
        alpha, trajectory = sample
        x0 = trajectory[0]
        axes_x = i % l_plot_grid
        axes_y = int(i / l_plot_grid)
        # axes[axes_x][axes_y].set_title(i, wrap=True)
        t0 = time.time()
        prediction = np.array(test(model, alpha, x0, len(trajectory) - 1))
        t_t.append(time.time() - t0)

        trajectory = np.array(trajectory)
        e_t.append(np.linalg.norm(prediction[1:] - trajectory[1:], 2) / (len(trajectory) - 1))

        for j in range(out_dim):
            color = scalar_map.to_rgba(j)
            axes[axes_x][axes_y].plot(trajectory[:, j].transpose(), color=color)
            axes[axes_x][axes_y].plot(prediction[:, j].transpose(), marker='o', color=color, markersize=3)

    if save_fig:
        fig_name = 'prediction_{}.png'.format(time.strftime("%H:%M:%S", time.localtime()))
        plt.savefig(fig_name, dpi=2000)
    plt.show()

    e = sum(e_t) / len(e_t)
    t = sum(t_t[1:]) / len(t_t[1:])
    logger.info('With model {}:'.format(name))
    logger.info('Average time consumption: {}s'.format(t))
    logger.info('Mean relative validation error: {}'.format(e))


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
