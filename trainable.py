import logging

import numpy as np
import torch
from torch import nn
from torchnet import meter

import utils.kits

logger = logging.getLogger(__name__)


def train(model, optimizer, train_loader, device=torch.device('cpu')):
    model.train()

    loss_meter = meter.AverageValueMeter()
    criterion = nn.MSELoss()

    for d in train_loader:
        x, y = d
        if x.size()[0] == 1:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if np.isnan(loss.item()):
            # print('NAN ALERT!')
            pass
        loss.backward()  # get gradients to parameters
        loss_meter.add(loss.data.cpu())
        optimizer.step()  # update parameters


def val(model, val_loader, device=torch.device('cpu'), dropout=False):
    model.eval()
    if dropout:
        # Keep the dropout layers working during prediction.
        # Dropout layers would be automatically turned off by PyTorch.
        model.apply(utils.kits.apply_dropout)

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


def test(model, alpha, x_0, length, dropout=True):
    model.eval()
    if dropout:
        # Keep the dropout layers working during prediction.
        # Dropout layers would be automatically turned off by PyTorch.
        model.apply(utils.kits.apply_dropout)

    prediction = [x_0]
    alpha = torch.tensor(alpha)
    x_0 = torch.tensor(x_0)
    state = torch.cat((x_0, alpha))

    with torch.no_grad():
        for i in range(length):
            x = model(state.unsqueeze(0)).squeeze(0)
            state = torch.cat((x, alpha))
            prediction.append(x.tolist())

    return prediction


def test_delta(model, alpha, trajectory, dropout=True):
    model.eval()
    if dropout:
        model.apply(utils.kits.apply_dropout)

    x_0 = trajectory[0][:-1]
    alpha = torch.tensor(alpha)
    prediction = [x_0]
    delta = torch.tensor(trajectory[0][-1]).reshape(1)
    state = torch.cat((torch.tensor(x_0).float(), alpha, delta))
    with torch.no_grad():
        for i in range(1, len(trajectory)):
            x = model(state.unsqueeze(0)).squeeze(0)
            state = torch.cat((x, alpha, torch.tensor(trajectory[i][-1]).reshape(1).float()))
            prediction.append(x.tolist())

    return prediction
