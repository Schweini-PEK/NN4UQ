"""
Here are some kits.
"""

import csv
import logging
import os
import random
from os.path import join, isfile

import numpy as np
import torch
from matplotlib import colors
from matplotlib import pyplot as plt
from torch import nn

logger = logging.getLogger(__name__)


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def mre(x_hat, x):
    """Calculate the mean relative error

    :param x_hat: The state vector prediction matrix: n_state * n_time_step
    :param x: The true state vector matrix
    :return: The mean relative validation error e_t
    """
    e_t = 0
    for i in range(len(x_hat)):
        e_t += np.linalg.norm(x_hat[i] - x[i], 2) / np.linalg.norm(x[i], 2)

    return e_t / len(x_hat)


def get_pth_from_dir(root):
    models_path = [join(root, f) for f in os.listdir(root) if (isfile(join(root, f)) and not f.startswith('.'))]
    return models_path


def get_pth_from_subdir(root):
    subs = [f.path for f in os.scandir(root) if f.is_dir()]
    models_path = []
    for sub in subs:
        files = os.listdir(sub)
        for file_name in files:
            if file_name.endswith('pth'):
                models_path.append(sub + '/' + file_name)

    return models_path


def columns2csv(output_path, *args):
    rows = zip(*args)
    rows2csv(output_path, rows)


def rows2csv(output_path, rows):
    with open(output_path, 'a+') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def color_wheel(num, theme='gist_rainbow'):
    c_map = plt.cm.get_cmap(theme)
    c_norm = colors.Normalize(vmin=0, vmax=num - 1)
    scalar_map = plt.cm.ScalarMappable(norm=c_norm, cmap=c_map)
    return scalar_map


def parse(conf, section, kwargs):
    for k, v in kwargs.items():
        if not conf.has_option(section, k):
            logger.warning('Opt has no attribute {}.'.format(k))
        else:
            conf.set(section, k, str(v))
            logger.info('Opt {} has been updated to {}'.format(k, v))


def _generate_notice(conf, section):
    notice = 'Using following parameters from section {}:'.format(section)
    for k, v in conf.items(section):
        notice += '\n{} {}'.format(k, v)
    return notice
