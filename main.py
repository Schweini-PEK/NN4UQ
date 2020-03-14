import warnings

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchnet import meter

import dataset
import models
import utils
from config import options

warnings.filterwarnings('ignore')


def train(**kwargs):
    options.parse(kwargs)

    model = getattr(models, options.model)()
    model.train()
    if options.use_gpu:
        model.cuda()

    vis = utils.Visualizer(options.env)
    data = dataset.lsode.LoadDataset(root=options.root, size=options.train_size)
    indices = list(range(len(data)))
    milestone = utils.kfold.k_fold_index_gen(indices, k=options.k_fold)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=options.lr)
    loss_meter = meter.AverageValueMeter()

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
            loss = 0
            for d in data_train_loader:
                optimizer.zero_grad()
                x, y = d
                if options.use_gpu:
                    x = x.cuda()
                    y = y.cuda()

                out = model(x.float())
                loss = criterion(out, y.float())
                loss.backward()
                optimizer.step()

                loss_meter.add(loss.data)

            if (epoch + 1) % options.print_freq == 0:
                # print('fold: {}, epoch: {}, loss: {:.4}'.format(fold + 1, epoch + 1, loss.data.item()))
                vis.plot('loss', loss_meter.value()[0])

        if len(milestone) != 2:  # Make sure there is a val dataset.
            loss_val = val(model, data_val_loader)
            vis.plot('val loss', loss_val.value()[0])

    options.load_model_path = model.save()


def val(model, data_loader: DataLoader):
    """

    :param model: The current model.
    :param data_loader: The validation dataset loader.
    :return:
    """
    model.eval()
    loss_meter = meter.AverageValueMeter()
    criterion = nn.MSELoss()

    for d in data_loader:
        x, y = d
        if options.use_gpu:
            x = x.cuda()
            y = y.cuda()

        out = model(x.float())
        loss = criterion(out, y.float())
        loss_meter.add(loss.data)

    return loss_meter


def predict(**kwargs):
    options.parse(kwargs)
    model = getattr(models, options.model)()
    if options.load_model_path:
        model.load(options.load_model_path)
    if options.use_gpu:
        model = model.cuda()

    vis = utils.Visualizer(options.env)

    x_init, alpha, delta = options.x_init, options.alpha, options.delta
    # state = torch.tensor([x_init, alpha, delta])
    state = torch.tensor([[x_init, alpha, delta], [x_init, alpha, delta]])
    x_truth = x_init

    # vis.plot('pred_1', float(state[0]))
    vis.plot('pred_1', float(state[0][0]))
    vis.plot('truth_1', x_truth)

    for i in range(options.trajectory_length):
        out = model(state.float())
        state[:, 0] = out.squeeze()
        # vis.plot('pred_1', float(state[0]))
        vis.plot('pred_1', float(state[0][0]))

        x_truth = utils.lsode.state_predictor(x_truth, alpha, delta)
        vis.plot('truth_1', x_truth)


def yitiaolong():
    train()
    predict()


if __name__ == '__main__':
    import fire

    fire.Fire()
