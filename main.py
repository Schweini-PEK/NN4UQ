import logging
import math
import pickle
import time

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

import utils
import utils.analysis
from config import config
from trainable import test

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


def forecast(save_fig=False):
    truth_path = 'dataset/NS_truth_x3a5.pkl'
    in_dim, out_dim = utils.kits.get_io_dim(truth_path)
    model_path = 'results/NS_x3a5/RSResNet_197_6_3.pth'
    model, legend = utils.kits.load_model_from_path(model_path, in_dim, out_dim)
    try:
        f = open(truth_path, 'rb')

    except FileNotFoundError:
        raise FileNotFoundError('No such file: {}'.format(config.truth_path))

    trajectories = pickle.load(f)
    l_plot_grid = math.ceil(math.sqrt(len(trajectories)))
    w_plot_grid = math.ceil(len(trajectories) / l_plot_grid)

    c_map = plt.cm.get_cmap('gist_rainbow')
    c_norm = colors.Normalize(vmin=0, vmax=out_dim - 1)
    scalar_map = plt.cm.ScalarMappable(norm=c_norm, cmap=c_map)
    fig, axes = plt.subplots(l_plot_grid, w_plot_grid, squeeze=False)
    fig.set_size_inches(18, 14)
    fig.suptitle('Predictions on {} sets of alpha'.format(len(trajectories)), y=0.02)

    for i, sample in enumerate(trajectories):
        alpha, trajectory = sample
        x0 = trajectory[0]
        axes_x = i % l_plot_grid
        axes_y = int(i / l_plot_grid)
        axes[axes_x][axes_y].set_title(alpha, wrap=True)
        prediction = np.array(test(model, alpha, x0, len(trajectory) - 1))
        trajectory = np.array(trajectory)

        for j in range(out_dim):
            color = scalar_map.to_rgba(j)
            axes[axes_x][axes_y].plot(trajectory[:, j].transpose(), color=color)
            axes[axes_x][axes_y].plot(prediction[:, j].transpose(), color=color, marker='o', markersize=3)

    if save_fig:
        fig = 'prediction_{}.png'.format(time.strftime("%H:%M:%S", time.localtime()))
        plt.savefig(fig, dpi=2000)
    plt.show()


if __name__ == '__main__':
    forecast(save_fig=False)
