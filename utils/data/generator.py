import csv
import logging
import os
import pickle
import random

import numpy as np

import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def data_generating(generate_func, n_data):
    return eval(generate_func)(n_data)


def lsode_generating(train_size, loop=1):
    """Generate the whole record set.

    :param loop: How many times the solver will run over on one x0.
    :param train_size: The number of samples.
    :return: A idx like [[x, alpha, delta], y]
    """

    dataset = []
    p_lower_bound = 0.0
    p_upper_bound = 1.0
    s_lower_bound = 0.1
    s_upper_bound = 0.8
    delta = 0.1

    for i in range(train_size):
        alpha = np.random.uniform(p_lower_bound, p_upper_bound)
        x_cur = np.random.uniform(s_lower_bound, s_upper_bound)
        for j in range(loop):
            x_next = utils.ode.ode_predictor(x_cur, alpha, delta)
            # dataset.append([[x_cur, alpha, delta], [x_next]])
            dataset.append([[x_cur, alpha, delta], [x_next]])
            x_cur = x_next

    return dataset


def mlode_generating(train_size):
    dataset = []
    d_lower_bound = 0.0
    d_upper_bound = 2.0
    delta = 0.1
    a = np.array([[1, -4], [4, -7]])

    for i in range(train_size):
        x = np.random.uniform(d_lower_bound, d_upper_bound, 2)
        x_next = utils.ode.multi_ode_predictor(x, a, delta)
        dataset.append([[x], [x_next]])

    return dataset


def _save(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        logger.info('{} record have been generated and saved at {}.'.format(len(data), path))


class Generator:
    def __init__(self, folder='dataset'):
        """

        :param folder: The folder to save dataset.
        """
        self.folder = folder
        try:
            os.mkdir(self.folder)
            logger.info('Folder {} created.'.format(self.folder))
        except FileExistsError as e:
            logger.info('{}: Folder {} already exists.'.format(e, self.folder))

    def generate(self, n_data, name=None, shuffle=True, method='lsode_generating', save=True):
        """

        :param shuffle: If shuffle.
        :param name: Name the file that is going to be saved later.
        :param n_data: The number of samples to be generated.
        :param method: The method to generate samples.
        :return: Data from a generating function or a csv file.
        """
        data = data_generating(method, n_data)
        path = self.folder + '/{}.pkl'.format(name if name
                                              else str(method).split('_')[0] + str(n_data))
        if shuffle:
            random.shuffle(data)
        if save:
            _save(data, path)
        return data

    def load_from_csv(self, x_path, name=None, shuffle=True, y_path=None, header=False, save=True):
        """Loading data from csv, with
        generator.load_from_csv(x_path='dataset/X.csv', y_path='dataset/Y.csv')

        :param shuffle: If shuffle.
        :param name: Name the file that is going to be saved later.
        :param save: If save.
        :param x_path: The path of data.
        :param y_path: The path of Y (if needed).
        :param header: True if the csv has a header.
        :return:
        """
        try:
            with open(x_path, 'r')as load_file:
                reader = csv.reader(load_file)
                if header:
                    next(reader)
                x = list(reader)

        except FileNotFoundError as e:
            logger.error('{}'.format(e))

        if y_path:  # If there is another file needs to be loaded.
            try:
                with open(y_path, 'r')as load_file:
                    reader = csv.reader(load_file)
                    if header:
                        next(reader)
                    y = list(reader)
                    x = list(map(list, zip(x, y)))

            except FileNotFoundError as e:
                logger.error('{}: File {} does not exists'.format(e, y_path))

        path = self.folder + '/{}.pkl'.format(name if name
                                              else x_path.split('/')[-1].split('.')[0])

        if shuffle:
            random.shuffle(x)
        if save:
            _save(x, path)
        return x
