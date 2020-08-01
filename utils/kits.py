"""
Here are some kits.
"""

import csv
import logging
import os
import random

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def get_pth_from_dir(root):
    models_path = [root + f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
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
