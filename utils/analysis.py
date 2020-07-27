import json
import math
import pickle
import time

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt

import utils
from trainable import test


def get_loss_from_ray(path):
    """Read and return a list of loss from 'result.json' generated by Ray.Tune

    :param path: The pat of 'result.json'
    :return: A list of loss
    """

    losses = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line_dict = json.loads(line)
            losses.append(line_dict['val_loss'])

    plt.plot(losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.show()
    return losses


def forecast(model_path, truth_path, save_fig=False):
    # torch.manual_seed(6)

    in_dim, out_dim = utils.kits.get_io_dim(truth_path)
    model, legend = utils.kits.load_model_from_path(model_path, in_dim, out_dim)

    try:
        f = open(truth_path, 'rb')
    except FileNotFoundError:
        raise FileNotFoundError('No such file: {}'.format(truth_path))

    trajectories = pickle.load(f)
    l_plot_grid = math.ceil(math.sqrt(len(trajectories)))
    w_plot_grid = math.ceil(len(trajectories) / l_plot_grid)

    c_map = plt.cm.get_cmap('gist_rainbow')
    c_norm = colors.Normalize(vmin=0, vmax=out_dim - 1)
    scalar_map = plt.cm.ScalarMappable(norm=c_norm, cmap=c_map)

    fig, axes = plt.subplots(l_plot_grid, w_plot_grid, squeeze=False, sharex=True, sharey=True)
    fig.text(0.5, 0.04, 'Time', ha='center')
    fig.text(0.04, 0.5, 'Quantities of Interest', va='center', rotation='vertical')
    # fig.suptitle('Predictions on {} sets of alpha'.format(len(trajectories)), y=0.02)

    for i, sample in enumerate(trajectories):
        alpha, trajectory = sample
        x0 = trajectory[0]
        axes_x = i % l_plot_grid
        axes_y = int(i / l_plot_grid)
        # axes[axes_x][axes_y].set_title(i, wrap=True)
        start = time.time()
        prediction = np.array(test(model, alpha, x0, len(trajectory) - 1))
        print(time.time() - start)
        trajectory = np.array(trajectory)

        for j in range(out_dim):
            color = scalar_map.to_rgba(j)
            axes[axes_x][axes_y].plot(trajectory[:, j].transpose(), color=color)
            axes[axes_x][axes_y].plot(prediction[:, j].transpose(), marker='o', color=color, markersize=3)

    if save_fig:
        fig_name = 'prediction_{}.png'.format(time.strftime("%H:%M:%S", time.localtime()))
        plt.savefig(fig_name, dpi=2000)
    plt.show()
