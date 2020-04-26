import logging
import time

import numpy as np
import torch
from ray.tune import Trainable
from torch import nn
from torchnet import meter

import utils

logger = logging.getLogger(__name__)


def train(model, optimizer, train_loader, device=torch.device('cpu')):
    model.train()

    loss_meter = meter.AverageValueMeter()
    criterion = nn.MSELoss()

    for d in train_loader:
        x, y = d
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x.float())
        loss = criterion(out, y.float())
        loss.backward()  # get gradients to parameters
        loss_meter.add(loss.data.cpu())
        optimizer.step()  # update parameters

    # return float(loss_meter.value()[0])


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

    acc = float(loss_meter.value()[0])
    return acc


def test(model, state, viz, win, name, dropout=True):
    model.eval()
    if dropout:
        # Keep the dropout layers working during prediction.
        # Dropout layers would be automatically turned off by PyTorch.
        model.apply(utils.kits.apply_dropout)

    length = state.get('length')
    x = state.get('x_0')
    alpha = state.get('alpha')
    x_model = torch.tensor(x + alpha)
    x_model_list = np.array([x_model[0]])

    with torch.no_grad():
        start_time = time.time()
        for i in range(length):
            x_model[0] = model(x_model.float())
            x_model_list = np.append(x_model_list, x_model[0])
        logger.info('{} finished predicting in {}'.format(name, time.time() - start_time))
        timeline = np.arange(0, length + 1)
        viz.line(X=timeline, Y=x_model_list, name=name, win=win, update='insert')


class Example(Trainable):
    def _setup(self, config):
        pass

    def _train(self):
        pass

    def _save(self, tmp_checkpoint_dir):
        pass

    def _restore(self, checkpoint):
        pass
