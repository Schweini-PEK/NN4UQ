import logging
import os
import time

import numpy as np
import pandas as pd

import models
import utils
from config import config


class Analysis:
    def __init__(self, grid=None, pretrained=False, path=None):
        self.name = time.strftime("%m%d_%H:%M/")
        if pretrained and path:
            self.df = pd.read_csv(path)
            self.name = path.split('/')[-1]
            for i, d in self.df.iterrows():
                self.df.at[i, 'train_loss'] = utils.kits.str2list(d['train_loss'])
                self.df.at[i, 'val_loss'] = utils.kits.str2list(d['val_loss'])
        elif grid:
            columns = list(grid.keys()) + ['train_loss', 'val_loss', 'name']
            self.df = pd.DataFrame(columns=columns)
            try:
                os.mkdir('checkpoints/' + self.name)
            except FileExistsError as e:
                logging.warning('{}: The folder already exists, the results might be confused.'.format(e))
        else:
            logging.error('No girds')

    def record(self, grid, record):
        """Record the results.

        :param grid:
        :param record:
        :return:
        """
        name = ''
        for k, v in grid.items():
            name = name + str(k) + '_' + str(v) + '_'
        grid.update(record)
        grid.update({'name': name})
        logging.info("{} lasted {}.".format(name, record['time']))
        self.df = self.df.append(grid, ignore_index=True)

    def save(self):
        """Save the result to csv file.
        analyst.save()

        :return: None
        """
        path = 'results/' + self.name[:-1] + '.csv'
        self.df.to_csv(path)
        logging.info('Experiment result has been saved to {}'.format(path))

    def _get_best_trials(self, n, idx='val_loss', method='avg'):
        """Sort the results by some standards. However, still haven't figured out the method.
        The avg of loss on validation set is too low to compare.

        :param n: Select the top n results.
        :param idx: The feature to use.
        :param method: Average.
        :return: The top n experiments.
        """
        if method == 'avg':
            def avg(x):
                return sum(x) / len(x)

            self.df[method] = self.df.apply(lambda row: avg(row[idx]))
            self.df = self.df.sort_values(method).drop(method, axis=1)
            return self.df.head(n)

    def plot_loss(self, viz):
        win_train, win_val = 'train', 'val'
        for i, d in self.df.iterrows():
            train_loss, val_loss = d['train_loss'], d['val_loss']
            train_timeline, val_timeline = np.arange(len(train_loss)), np.arange(len(val_loss))
            name = d['name']
            print(d['val_loss'])
            if i == 1:
                viz.line(win=win_train, name=name, Y=train_loss, X=train_timeline,
                         opts=dict(title=win_train, legend=[name], showlegend=True))
                viz.line(win=win_val, name=name, Y=val_loss, X=val_timeline,
                         opts=dict(title=win_val, legend=[name], showlegend=True))
            else:
                viz.line(win=win_train, name=name, Y=train_loss, X=train_timeline, update='insert')
                viz.line(win=win_val, name=name, Y=val_loss, X=val_timeline, update='insert')

    def predict(self, viz, state, func, use_best=False, n=1, idx='val_loss',
                in_dim=3, out_dim=1):
        """Use the trained model to predict.
        TODO: Make the function more generalized for the changes of i/o dimensions.

        :param out_dim: The input dimension of the model.
        :param in_dim: The output dimension of the model.
        :param viz: Visdom.
        :param state: The initial state to be predicted.
        :param func: The test function.
        :param use_best: Just use the best ones to predict. Otherwise all the models would be used.
        :param n: Picking the top n results sorted by idx.
        :param idx: Sorting the results by idx.
        :return: None
        """
        t = 0.0
        length = 300
        x_solver = 1.0
        delta = 0.1
        alpha = 0.3225
        # length = state.get('length', 300)
        # x_solver = state.get('x_0', 1.0)
        # delta = state.get('delta', 0.1)
        # alpha = state.get('alpha')
        x_solver_list = np.array([x_solver])
        win = self.name

        for i in range(length):
            t += delta
            x_solver = utils.ode.ode_predictor(x_solver, alpha, delta)
            x_solver_list = np.append(x_solver_list, x_solver)
        timeline = np.arange(0, length + 1) * delta
        viz.line(X=timeline, Y=x_solver_list,
                 name='solver', win=win,
                 opts=dict(title='predict_' + self.name, legend=['solver'], showlegend=True)
                 )

        df = self._get_best_trials(n, idx) if use_best else self.df

        for _, grid in df.iterrows():
            if grid['model'] == 'RSResNet' or 'RTResNet':
                model = getattr(models, grid['model'])(in_dim=in_dim, out_dim=out_dim, k=config.k)
            else:
                model = getattr(models, grid['model'])(in_dim=in_dim, out_dim=out_dim)
            model.load(grid['ckpt'])
            logging.info('Trajectory plotted with model {}'.format(grid['name']))
            func(model, state, viz, win=win, name=grid['name'], dropout=False)

        # truth_path = 'dataset/trajectory.pkl'
        # try:
        #     with open(truth_path, 'rb') as f:
        #         x_truth = pickle.load(f)
        #         viz.line(X=timeline[:-1], Y=np.array(x_truth),
        #                  win=win, name='truth',
        #                  update='insert')
        # except FileNotFoundError:
        #     logging.warning('No ground truth available at {}.'.format(truth_path))
