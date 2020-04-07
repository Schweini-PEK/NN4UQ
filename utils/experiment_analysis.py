import os
import time

import numpy as np
import pandas as pd

import models
import utils


class Analysis:
    def __init__(self, grid=None, pretrained=False, path=None):
        self.name = time.strftime("%m%d_%H:%M/")
        if pretrained and path:
            self.df = pd.read_csv(path)
            for i, d in self.df.iterrows():
                self.df.at[i, 'train_loss'] = utils.kits.str2list(d['train_loss'])
                self.df.at[i, 'val_loss'] = utils.kits.str2list(d['val_loss'])
        elif grid:
            columns = list(grid.keys()) + ['train_loss', 'val_loss', 'name']
            self.df = pd.DataFrame(columns=columns)
            try:
                os.mkdir('checkpoints/' + self.name)
            except FileExistsError as e:
                print('{}: The folder already exists, the results might be confused.'.format(e))
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
        self.df.to_csv('results/' + self.name[:-1] + '.csv')

    def _get_best_trials(self, n, idx, method='avg'):
        if method == 'avg':
            def avg(x):
                return sum(x) / len(x)

            self.df[method] = self.df.apply(lambda row: avg(row[idx]))
            self.df = self.df.sort_values(method).drop(method, axis=1)
            return self.df.head(n)

    def plot(self, viz):
        win_train, win_val = 'train', 'val'
        for _, d in self.df.iterrows():
            train_loss, val_loss = d['train_loss'], d['val_loss']
            train_timeline, val_timeline = np.arange(len(train_loss)), np.arange(len(val_loss))
            name = d['name']
            viz.line(win=win_train, name=name, Y=train_loss, X=train_timeline, update='append')
            viz.line(win=win_val, name=name, Y=val_loss, X=val_timeline, update='append')

    def predict(self, viz, state, func, dp=False, use_best=False, n=1, idx='val_loss'):
        """

        :param dp: Using dropout during predicting.
        :param viz: Visualizer.
        :param state: The initial state to be predicted.
        :param func: The test function.
        :param use_best: Just use the best ones to predict. Otherwise all the models would be used.
        :param n: Picking the top n results sorted by idx.
        :param idx: Sorting the results by idx.
        :return:
        """
        df = self._get_best_trials(n, idx) if use_best else self.df

        for _, grid in df.iterrows():
            model = getattr(models, grid['model'])()
            model.load(grid['ckpt'])
            func(model, state, viz, win=grid['name'], dropout=dp)
