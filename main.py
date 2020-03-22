import pickle
import warnings

import numpy as np
import torch
import visdom
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchnet import meter

import dataset
import models
import utils
from config import options

warnings.filterwarnings('ignore')


def generate(**kwargs):
    options.parse(kwargs)

    n_data = options.n_data
    math_model = options.math_model
    generating_method = math_model + "_generating"

    training_data = dataset.data_generating(generating_method, n_data)
    path = options.root + math_model + str(n_data) + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(training_data, f)


def train(**kwargs):
    options.parse(kwargs)

    model = getattr(models, options.model)()
    model.train()
    if options.use_gpu:
        model.cuda()

    viz = visdom.Visdom()
    data = dataset.loader.LoadDataset(root=options.root, size=options.n_data)
    indices = list(range(len(data)))
    milestone = utils.kfold.k_fold_index_gen(indices, k=options.k_fold)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())
    loss_meter = meter.AverageValueMeter()
    loss_list = []

    for fold in range(len(milestone) - 1):
        if len(milestone) == 2:  # when k_fold is not just ONE FOLD.
            train_indices = indices
            val_indices = []
        else:
            train_indices = indices[:milestone[fold]] + indices[milestone[fold + 1]:]
            val_indices = indices[milestone[fold]:milestone[fold + 1]]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        data_train_loader = DataLoader(data, batch_size=options.batch_size,
                                       num_workers=options.num_workers, sampler=train_sampler)
        data_val_loader = DataLoader(data, batch_size=options.batch_size,
                                     num_workers=options.num_workers, sampler=val_sampler)

        for epoch in range(options.max_epoch):
            loss_meter.reset()
            for d in data_train_loader:
                optimizer.zero_grad()
                x, y = d
                if options.use_gpu:
                    x = x.cuda()
                    y = y.cuda()

                out = model(x.float())
                loss = criterion(out, y.float())

                loss.backward()  # get gradients to parameters
                optimizer.step()  # update parameters
                loss_meter.add(loss.data)

            if (epoch + 1) % 500 == 0:
                model.save()

        if len(milestone) != 2:  # Make sure there is a val dataset.
            val(model, data_val_loader)

    viz.line(Y=np.asarray(loss_list), X=np.asarray(range(options.max_epoch * options.k_fold)))

    model.save()


def val(model, data_loader: DataLoader):
    """

    :param model: The current model.
    :param data_loader: The validation dataset loader.
    :return:
    """
    model.eval()
    loss_meter = meter.AverageValueMeter()
    criterion = nn.MSELoss()
    viz = visdom.Visdom()

    with torch.no_grad():
        for d in data_loader:
            x, y = d
            if options.use_gpu:
                x = x.cuda()
                y = y.cuda()

            out = model(x.float())
            loss = criterion(out, y.float())
            loss_meter.add(loss.data)
            # vis.plot('val loss', loss_meter.value()[0])
            viz.line(Y=loss_meter.value()[0], update="append")

    return loss_meter


def predict(**kwargs):
    options.parse(kwargs)
    model = getattr(models, options.model)()
    model.load(options.load_model_path)
    if options.use_gpu:
        model = model.cuda()

    x_init, alpha, delta = options.x_init, options.alpha, options.delta
    state = torch.tensor([x_init, alpha, delta])
    x_truth = x_init
    time = 0.0

    viz = visdom.Visdom()
    win = viz.line(
        X=np.column_stack((time, time)),
        Y=np.column_stack((state[0], x_truth))
    )

    with torch.no_grad():
        for i in range(options.trajectory_length):
            time += delta
            state[0] = model(state.float())
            print("state", state[0], state[1])

            x_truth = utils.ode.ode_predictor(x_truth, alpha, delta)
            viz.line(
                X=np.column_stack((time, time)),
                Y=np.column_stack((state[0], x_truth)),
                win=win,
                update="append"
            )


def grid_train(**kwargs):
    options.parse(kwargs)

    viz = visdom.Visdom()
    model = getattr(models, options.model)()
    if options.use_gpu:
        model = model.cuda()

    data = dataset.loader.LoadDataset(root=options.root, size=options.n_data)
    net = NeuralNetRegressor(models.MLP, max_epochs=options.max_epoch, lr=options.lr, verbose=1)
    n_features = len(data.ode_frame[0][0])
    x, y = utils.tools.list2sample(n_features, data.ode_frame)
    x = torch.tensor(x)
    y = torch.tensor(y)

    params = {'lr': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
              'max_epochs': list(range(200, 2200, 200))}

    gs = GridSearchCV(net, params, scoring='neg_mean_absolute_error',
                      verbose=1, cv=5, n_jobs=4)

    gs.fit(x.float(), y.float())
    gs.get_params()
    print("my gs: ", gs.best_score_, gs.best_params_)

    utils.tools.print_best_score(gs, params)


def rnn_train(**kwargs):
    options.parse(kwargs)

    viz = visdom.Visdom()
    model = getattr()


if __name__ == '__main__':
    import fire

    fire.Fire()
