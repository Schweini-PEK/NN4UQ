import inspect
import pickle
import time
import warnings

import numpy as np
import torch
from ray import tune
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
from torch import nn, optim
from torch.utils.data import DataLoader
from torchnet import meter

import dataset
import models
import utils
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


def grid_train(**kwargs):
    config.parse(kwargs)

    data = dataset.loader.LoadDataset(math_model=config.math_model, size=config.n_data)
    net = NeuralNetRegressor(models.MLP, max_epochs=config.max_epoch, lr=config.lr, verbose=1)
    n_features = len(data.ode_frame[0][0])
    x, y = utils.kits.list2sample(n_features, data.ode_frame)
    x = torch.tensor(x)
    y = torch.tensor(y)

    params = {'lr': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
              'max_epochs': list(range(1000, 2200, 200))}

    gs = GridSearchCV(net, params, scoring='neg_mean_absolute_error',
                      verbose=1, cv=5, n_jobs=4)

    gs.fit(x.float(), y.float())
    gs.get_params()
    # print("my gs: ", gs.best_score_, gs.best_params_)

    utils.kits.print_best_score(gs, params)


# def general(**kwargs):
def general(search_space):
    # config.parse(kwargs)
    # viz = utils.Visualizer()
    # win = inspect.currentframe().f_code.co_name + '@' + time.strftime("%H:%M:%S", time.localtime())

    use_cuda = config.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = getattr(models, search_space['model'])().to(device)
    optimizer = optim.Adam(model.parameters(), lr=search_space['lr'])

    data = dataset.loader.LoadDataset(math_model=config.math_model, size=config.n_data,
                                      base=config.base)
    train_loader, test_loader = utils.kits.get_data_loaders(data, config.train_ratio,
                                                            config.batch_size, config.num_workers)
    func = config.func
    if func == "train":
        print(search_space['max_epoch'])
        for i in range(search_space['max_epoch']):
            train(model, optimizer, train_loader, device)
            acc = val(model, test_loader, device)
            tune.track.log(mean_accuracy=acc)

            # if config.visible:
            #     if (i + 1) % config.print_freq == 0:
            #         viz.plot(win=win, name='loss', y=loss[-1])

            if (i + 1) % 500 == 0:
                model.save()

    elif func == "predict":
        # predict(model, config, viz)
        pass


def train(model, optimizer, train_loader, device=torch.device('cpu')):
    model.train()

    criterion = nn.L1Loss()
    loss_meter = meter.AverageValueMeter()

    for d in train_loader:
        optimizer.zero_grad()
        x, y = d
        x, y = x.to(device), y.to(device)

        out = model(x.float())
        loss = criterion(out, y.unsqueeze(1).float())
        loss.backward()  # get gradients to parameters
        optimizer.step()  # update parameters
        loss_meter.add(loss.data)
    return loss_meter.value()[0]


def val(model, data_loader: DataLoader, device=torch.device('cpu')):
    model.eval()

    loss_meter = meter.AverageValueMeter()
    criterion = nn.MSELoss()

    with torch.no_grad():
        for d in data_loader:
            x, y = d
            x, y = x.to(device), y.to(device)

            out = model(x.float())
            loss = criterion(out, y.unsqueeze(1).float())
            loss_meter.add(loss.data)

    return loss_meter.value()[0]


def ray_train(**kwargs):
    config.parse(kwargs)
    search_space = {
        'lr': tune.choice([0.001, 0.01, 0.1, 0.2, 0.3]),
        'max_epoch': tune.choice([100, 500, 1000, 2000]),
        'model': tune.choice(['MLP'])
    }
    resource = None
    if config.use_gpu and torch.cuda.is_available():
        resource = {'gpu': 1}
        print("using GPU")
    analysis = tune.run(general, resources_per_trial=resource, config=search_space)
    df = analysis.dataframe()

    dfs = analysis.trial_dataframes
    ax = None
    print(len(dfs.values()))
    for d in dfs.values():
        print('D: ', d)
        ax = d.mean_accuracy.plot(ax=ax, legend=False)

    # logdir = analysis.get_best_logdir("mean_accuracy", mode="max")
    # model = torch.load(os.path.join(logdir, "model.pth"))
    # print(model.parameters())


def predict(model, cfg, viz: utils.Visualizer):
    model.eval()
    length = cfg.trajectory_length
    t = 0.0
    state = cfg.predicting
    if state['mission'] == "single-trajectory":
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
