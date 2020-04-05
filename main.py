import collections
import inspect
import pickle
import time
import warnings

import numpy as np
import torch
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


def general(grids):
    use_cuda = config.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    data = dataset.loader.LoadDataset(math_model=config.math_model, size=config.n_data)
    train_loader, test_loader = utils.kits.get_data_loaders(data, config.train_ratio,
                                                            config.batch_size, config.num_workers)

    analyst = utils.experiment_analysis.Analysis(grids[0])

    for grid in grids:
        t0 = time.time()
        model = getattr(models, grid['model'])().to(device)
        optimizer = optim.Adam(model.parameters(), lr=grid['lr'])

        train_loss_list, val_loss_list = [], []
        for i in range(grid['max_epoch']):
            train_loss = train(model, optimizer, train_loader, device)
            train_loss_list.append(train_loss)

            if (i + 1) % 5 == 0:
                val_loss = val(model, test_loader, device)
                val_loss_list.append(val_loss)

        t = time.time() - t0
        checkpoint = model.save()
        data = {'train_loss': train_loss_list, 'val_loss': val_loss_list,
                'ckpt': checkpoint, 'time': t}
        analyst.record(grid, data)

    analyst.save()
    return analyst


def train(model, optimizer, train_loader, device=torch.device('cpu')):
    model.train()

    loss_meter = meter.AverageValueMeter()
    criterion = nn.L1Loss()

    for d in train_loader:
        optimizer.zero_grad()
        x, y = d
        x, y = x.to(device), y.to(device)

        out = model(x.float())
        loss = criterion(out, y.unsqueeze(1).float())
        loss.backward()  # get gradients to parameters
        loss_meter.add(loss.data.cpu())
        optimizer.step()  # update parameters

    return loss_meter.value()[0]


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

    return loss_meter.value()[0]


def tune_train(**kwargs):
    config.parse(kwargs)
    params = {'model': ['MLP', 'resnet', 'ShallowResBN', 'rs_resnet'],
              'lr': [0.05, 0.1, 0.2],
              'max_epoch': [1000]}

    params = collections.OrderedDict(sorted(params.items()))
    grids = list(ParameterGrid(params))

    analyst = general(grids)


def plotter():
    path = 'results/' + config.load_result_path + '.csv'
    analyst = utils.experiment_analysis.Analysis(pretrained=True, path=path)
    viz = utils.Visualizer()
    analyst.plot(viz)
    analyst.predict(viz, config.state, test)


def test(model, state, viz, truth_path="dataset/trajectory.pkl"):
    model.eval()
    t = 0.0
    length = state.get('length', 300)
    x_solver = x_0 = state.get('x_0', 1.0)
    delta = state.get('delta', 0.1)
    alpha = state.get('alpha')
    x_model = torch.tensor([x_0, alpha, delta])
    win = inspect.currentframe().f_code.co_name + '@' + time.strftime("%H:%M:%S", time.localtime())

    viz.plot(win=win, name='model', y=float(x_model[0]), x=t)
    viz.plot(win=win, name='solver', y=x_solver, x=t)
    with torch.no_grad():
        for i in range(length):
            t += delta
            x_model[0] = model(x_model.float())
            x_solver = utils.ode.ode_predictor(x_solver, alpha, delta)
            viz.plot(win=win, name='model', y=float(x_model[0]), x=t)
            viz.plot(win=win, name='solver', y=x_solver, x=t)
    try:
        with open(state.get('truth_path'), 'rb') as f:
            x_truth = pickle.load(f)
            timeline = np.arange(len(x_truth)) * delta
            viz.plot(win=win, name='truth', y=x_truth, x=timeline)
    except FileNotFoundError:
        print('No ground truth available.')


if __name__ == '__main__':
    import fire

    fire.Fire()
