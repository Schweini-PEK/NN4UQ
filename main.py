"""There are some things you have to do before getting through these scripts.
!pip install -r requirements.txt
!python -m visdom.server

To use tune train, run:
!python main.py tune_train

The configures in config.py could be changed temporarily with commands in the following way:
!python main.py tune_train

To load data from .csv:
data_generator = generator.Generator()
data = data_generator.load_from_csv(x_path='dataset/X.csv', y_path='dataset/Y.csv', save=True, shuffle=False)

"""

import collections
import logging
import time

import numpy as np
import torch
import visdom
from sklearn.model_selection import ParameterGrid
from torch import nn, optim
from torchnet import meter

import models
import utils
import utils.experiment_analysis
from config import config
from utils.data import loader

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


def trainable(grids, data=None):
    # GPU
    use_cuda = config.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info('Using {}'.format(device))

    # Initialize analyst
    analyst = utils.experiment_analysis.Analysis(grids[0])
    init_time = time.time()
    result_folder = analyst.name

    # Go through every grid
    for grid in grids:
        config.parse(grid)  # Update grid into config
        logger.info('Using config: {}'.format(grid))

        data = utils.data.loader.LoadDataset(path=config.dp, sample_freq=config.sf)
        in_dim, out_dim = len(data[0][0]), len(data[0][1])
        logger.info('The input and output dimensions of models are {}, {}'.format(in_dim, out_dim))
        train_loader, test_loader = utils.data.loader.get_data_loaders(data, config.bs,
                                                                       config.train_ratio, num_workers=4)

        if config.model == 'RSResNet':
            model = models.RSResNet(in_dim=in_dim, out_dim=out_dim, k=config.k).to(device)
        else:
            # If you want to tune the number of nodes in hidden layers, here is an easier way to do so.
            # model = getattr(models, config.model)(in_dim=in_dim, n_hidden=config.nh out_dim=out_dim).to(device)
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


def train(model, optimizer, train_loader, device=torch.device('cpu')):
    model.train()

    loss_meter = meter.AverageValueMeter()
    criterion = nn.MSELoss()

    for d in train_loader:
        optimizer.zero_grad()
        x, y = d
        x, y = x.to(device), y.to(device)

        out = model(x.float())
        loss = criterion(out, y.float())
        loss.backward()  # get gradients to parameters
        loss_meter.add(loss.data.cpu())
        optimizer.step()  # update parameters

    return float(loss_meter.value()[0])


def val(model, val_loader, device=torch.device('cpu')):
    model.eval()

    loss_meter = meter.AverageValueMeter()
    criterion = nn.MSELoss()

    with torch.no_grad():
        for d in val_loader:
            x, y = d
            x, y = x.to(device), y.to(device)

            out = model(x.float())
            loss = criterion(out, y.float())
            loss_meter.add(loss.data.cpu())

    return float(loss_meter.value()[0])


def tune_train(**kwargs):
    config.parse(kwargs)
    # Change the things here for the hyperparameter tuning.
    # Some other parts are also available to change.
    # i.e., to use different data in a single experiments, simply add:
    # 'dp': ['dataset/you_name_it', ...]
    # The name has to be the same with those in config.py
    # params = {'model': ['RSResNet'],  # The name of the model.
    params = {'model': ['RSResNet'],  # The name of the model.
              'lr': [0.03],
              'epoch': [140],
              'bs': [4],
              'sf': [100, 60, 10]
              }  # Batch Size

    params = collections.OrderedDict(sorted(params.items()))
    grids = list(ParameterGrid(params))

    analyst = trainable(grids)
    viz = visdom.Visdom()
    analyst.plot_loss(viz)
    analyst.predict(viz, config.state, test)


def plotter():
    """ This is a function to recall the results of the experiments.

    :return:
    """
    load_result_path = 'results/0416_01:55.csv'
    analyst = utils.experiment_analysis.Analysis(pretrained=True, path=load_result_path)
    viz = visdom.Visdom()
    analyst.plot_loss(viz)
    # Change the dimensions of models if needed. This would be improved to be more automatically.
    # See more at the declaration.
    analyst.predict(viz, config.state, test)


def test(model, state, viz, win, name, dropout=True):
    model.eval()
    if dropout:
        # Keep the dropout layers working during prediction.
        # Dropout layers would be automatically turned off by PyTorch.
        model.apply(utils.kits.apply_dropout)

    t = 0.0
    length = state.get('length', 300)
    x_solver = state.get('x_0', 1.0)
    delta = state.get('delta', 0.1)
    alpha = state.get('alpha', 0.5)
    x_model = torch.tensor([x_solver, alpha, delta])
    x_model_list = np.array([x_model[0]])
    x_solver_list = np.array([x_solver])

    with torch.no_grad():
        for i in range(length):
            t += delta
            x_model[0] = model(x_model.float())
            x_solver = utils.ode.ode_predictor(x_solver, alpha, delta)
            x_model_list = np.append(x_model_list, x_model[0])
            x_solver_list = np.append(x_solver_list, x_solver)
        timeline = np.arange(0, length + 1) * delta
        viz.line(X=timeline, Y=x_model_list, name=name, win=win, update='insert')


if __name__ == '__main__':
    import fire

    fire.Fire()
