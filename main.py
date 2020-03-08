import warnings

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
    data = dataset.lsode.LoadDataset(root=options.root)  # enter 'size=' to change the dataset (have to be existed).
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
                print('fold: {}, epoch: {}, loss: {:.4}'.format(fold + 1, epoch + 1, loss.data.item()))
                vis.plot('loss', loss_meter.value()[0])

        if len(milestone) != 2:  # Make sure there is a val dataset.
            loss_val = val(model, data_val_loader)
            vis.plot('val loss', loss_val.value()[0])

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

    for d in data_loader:
        x, y = d
        if options.use_gpu:
            x = x.cuda()
            y = y.cuda()

        out = model(x.float())
        loss = criterion(out, y.float())
        loss_meter.add(loss.data)

    return loss_meter


def test(**kwargs):
    options.parse(kwargs)
    model = getattr(models, options.model)()
    if options.load_model_path:
        model.load(options.load_model_path)
    if options.use_gpu:
        model = model.cuda()

    test_data = dataset.lsode.LoadDataset(test=True, root=options.root + options.load_testset_path)
    data_test_loader = DataLoader(test_data, num_workers=options.num_workers)
    for d in data_test_loader:
        x, y = d
        # TODO finish the prediction


if __name__ == '__main__':
    import fire

    fire.Fire()
