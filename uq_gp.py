import logging

import torch
import torch.optim as optim
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

import models
import utils
from uq_toy import train, val
from utils.data import loader
from utils.kits import setup_seed

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(module)s - %(message)s')
SEED = 42

# Define search space and initial guess
dim_learning_rate = Real(low=1e-3, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_num_dense_layers = Integer(low=2, high=6, name='num_layers')
dim_num_dense_nodes = Integer(low=4, high=15, name='num_nodes')

dim_batch_size = Integer(low=5, high=20, name='batch_size')
dim_epochs = Integer(low=10, high=20, prior='log-uniform', name='epochs')
dim_blocks = Integer(low=2, high=4, name='blocks')

dimensions = [dim_learning_rate,
              dim_batch_size,
              dim_epochs,
              dim_blocks,
              dim_num_dense_layers,
              dim_num_dense_nodes
              ]

default_parameters = [3e-3, 10, 15, 3, 2, 4]


@use_named_args(dimensions=dimensions)
def loss_func(learning_rate, batch_size, epochs, blocks, num_layers, num_nodes):
    # Call the following func to set up seed.
    setup_seed(SEED)
    path = 'dataset/george_ns.pkl'
    data = utils.data.loader.LoadDataset(path=path)

    train_loader, test_loader = utils.data.loader.get_data_loaders(data,
                                                                   batch_size=bs,
                                                                   num_workers=4)

    print('Using lr={}, bs={}, epoch={}, k={}, n_l={}, n_n={}'.
          format(learning_rate, batch_size, epochs, blocks, num_layers, num_nodes))

    in_dim, out_dim = len(data[0][0]), len(data[0][1])
    print('The input and output dimensions of models are {}, {}'.format(in_dim, out_dim))
    model = models.RSResNet(in_dim=in_dim, out_dim=out_dim, k=blocks,
                            h_dim=num_nodes, n_h_layers=num_layers)

    # print the structure of NN.
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate)
    for i in range(epochs):
        train(model, optimizer, train_loader, torch.device("cpu"))
        acc = val(model, test_loader, torch.device("cpu"))
        print('{} epoch: acc={}'.format(i, acc))
    name = str(learning_rate) + str(batch_size) + str(epochs)
    model.save(path="GP_results/{}.pth".format(name))
    print('GP iteration with acc:', acc)
    return acc


if __name__ == '__main__':
    gp_result = gp_minimize(func=loss_func,
                            dimensions=dimensions,
                            n_calls=11,
                            noise=1e-8,
                            n_jobs=-1,
                            acq_func="EI",
                            x0=default_parameters)
