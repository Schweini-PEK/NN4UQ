import collections
import pickle
import time
import warnings

import numpy as np
import torch
import visdom
from sklearn.model_selection import ParameterGrid
from torch import nn, optim
from torchnet import meter

import dataset
import models
import utils
import utils.experiment_analysis
from config import config

warnings.filterwarnings('ignore')


def generate(**kwargs):
    config.parse(kwargs)

    n_data = config.n_data
    math_model = config.math_model
    generating_method = math_model + "_generating"

    training_data = dataset.data_generating(generating_method, n_data)
    path = config.root + math_model + str(n_data) + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(training_data, f)


def trainable(grids):
    use_cuda = config.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    analyst = utils.experiment_analysis.Analysis(grids[0])
    folder = analyst.name

    for grid in grids:
        data = dataset.loader.LoadDataset(math_model=config.math_model, size=config.n_data)
        train_loader, test_loader = utils.kits.get_data_loaders(data, config.train_ratio,
                                                                grid['bs'], num_workers=4)
        t0 = time.time()
        model = getattr(models, grid['model'])().to(device)
        optimizer = optim.SGD(model.parameters(), lr=grid['lr'])

        train_loss_list, val_loss_list = [], []
        for i in range(grid['epoch']):
            train_loss = train(model, optimizer, train_loader, device)
            train_loss_list.append(train_loss)

            if (i + 1) % 5 == 0:
                val_loss = val(model, test_loader, device)
                val_loss_list.append(val_loss)

            if (i + 1) % 50 == 0:
                print('{i} epoch with {time}s'.format(i=i + 1, time=time.time() - t0))

        t = time.time() - t0
        checkpoint = model.save(folder=folder)
        data = {'train_loss': train_loss_list, 'val_loss': val_loss_list,
                'ckpt': checkpoint, 'time': t}
        analyst.record(grid, data)

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
        loss = criterion(out, y.unsqueeze(1).float())
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
            loss = criterion(out, y.unsqueeze(1).float())
            loss_meter.add(loss.data.cpu())

    return float(loss_meter.value()[0])


def tune_train(**kwargs):
    config.parse(kwargs)
    params = {'model': ['ResNet'],
              'lr': [0.05, 0.1, 0.2],
              'epoch': [600],
              'bs': [4, 8]}

    params = collections.OrderedDict(sorted(params.items()))
    grids = list(ParameterGrid(params))

    analyst = trainable(grids)
    viz = visdom.Visdom()
    analyst.plot(viz)
    analyst.predict(viz, config.state, test)


def plotter():
    path = 'results/0407_12:00.csv'
    analyst = utils.experiment_analysis.Analysis(pretrained=True, path=path)
    viz = visdom.Visdom()
    analyst.plot(viz)
    analyst.predict(viz, config.state, test, dp=False)


def test(model, state, viz, win, dropout, truth_path="dataset/trajectory.pkl"):
    model.eval()

    if dropout:
        model.apply(utils.kits.apply_dropout)

    t = 0.0
    length = state.get('length', 300)
    x_solver = state.get('x_0', 1.0)
    delta = state.get('delta', 0.1)
    alpha = state.get('alpha')
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
        viz.line(X=timeline, Y=x_model_list, name='model', win=win, opts=dict(title=win))
        viz.line(X=timeline, Y=x_solver_list, name='solver', win=win, update='insert')
    try:
        with open(truth_path, 'rb') as f:
            x_truth = pickle.load(f)
            viz.line(X=timeline[:-1], Y=np.array(x_truth), win=win, name='truth', update='insert')
    except FileNotFoundError:
        print('No ground truth available.')


if __name__ == '__main__':
    import fire

    fire.Fire()
