import logging
import time

import numpy as np
import pandas as pd

import models

logger = logging.getLogger(__name__)


class Analysis:
    def __init__(self, grid=None, pretrained=False, path=None):
        if pretrained and path:
            self.df = pd.read_csv(path)
        elif grid:
            columns = list(grid.keys()) + ['train_loss', 'val_loss', 'name']
            self.df = pd.DataFrame(columns=columns)
        else:
            raise ValueError('No grids.')

    def record(self, grid, data):
        name = ''
        for k, v in grid.items():
            name = name + str(k) + '_' + str(v) + '_'
        grid.update(data)
        grid.update({'name': name})
        print("{model} finished within {time}.".format(model=name, time=data['time']))
        self.df = self.df.append(grid, ignore_index=True)

    def save(self):
        self.df.to_csv('results/' + time.strftime("%H:%M:%S", time.localtime()) + '.csv')

    def _get_best_trials(self, n, idx):
        return self.df.sort_values(by=[idx]).head(n)

    def plot(self, viz):
        win_train, win_val = 'train', 'val'
        for _, d in self.df.iterrows():
            train_loss = d['train_loss']
            train_timeline = np.arange(len(train_loss))
            val_loss = d['val_loss']
            val_timeline = np.arange(len(val_loss))
            name = d['name']
            viz.plot(win=win_train, name=name, y=train_loss, x=train_timeline)
            viz.plot(win=win_val, name=name, y=val_loss, x=val_timeline)

    def predict(self, viz, state, func, use_best=True, n=1, idx='val_loss'):
        """

        :param viz: Visualizer.
        :param state: The initial state to be predicted.
        :param func: The test function.
        :param use_best: Just use the best ones to predict. Otherwise all the models would be used.
        :param n: Picking the top n results sorted by idx.
        :param idx: Sorting the results by idx.
        :return:
        """
        dfs = self._get_best_trials(n, idx) if use_best else self.df

        for _, grid in dfs.iterrows():
            model = getattr(models, grid['model'])()
            model.load(grid['ckpt'])
            func(model, state, viz)
