import logging
import os
import random

import torch
import torch.optim as optim
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from torch.utils.data import DataLoader

import models
import utils
from trainable import train, val
from utils.data import loader
from utils.kits import setup_seed

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(module)s - %(message)s')
SEED = random.randint(11, 25)
print('SEED: {}'.format(SEED))

# Define search space and initial guess
dim_learning_rate = Real(low=1e-3, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_batch_size = Integer(low=4, high=20, name='batch_size')
# dim_epochs = Integer(low=400, high=700, prior='log-uniform', name='epochs')
# dim_blocks = Integer(low=2, high=4, name='blocks')
# num_nodes ** 2 * (num_layers - 1)
dim_num_dense_layers = Integer(low=2, high=5, name='num_layers')
dim_num_dense_nodes = Integer(low=20, high=60, name='num_nodes')

dimensions = [dim_learning_rate,
              dim_batch_size,
              # dim_epochs,
              # dim_blocks,
              dim_num_dense_layers,
              dim_num_dense_nodes
              ]

# default_parameters = [9e-3, 40, 400, 3, 7, 20]
default_parameters = [9e-3, 12, 2, 20]


@use_named_args(dimensions=dimensions)
def loss_func(learning_rate, batch_size, num_layers, num_nodes):
    # Call the following func to set up seed.
    root = os.path.dirname(os.path.realpath(__file__))
    setup_seed(SEED)
    path = root + '/dataset/300_reals_x3a6_1000.pkl'
    #  3,000 samples in the .pkl file
    data = utils.data.loader.LoadDataset(path=path)
    train_loader, test_loader = utils.data.loader.get_data_loaders(data,
                                                                   batch_size=int(batch_size),
                                                                   num_workers=4)

    data_2 = utils.data.loader.LoadDataset(path=root + '/dataset/50_val_x3a6_1177.pkl')
    test_loader_2 = DataLoader(data_2, batch_size=1, num_workers=4, shuffle=True)
    print('Using lr={}, bs={}, n_l={}, n_n={}'.
          format(learning_rate, batch_size, num_layers, num_nodes))

    in_dim, out_dim = len(data[0][0]), len(data[0][1])
    model = models.ResNet(in_dim=in_dim, out_dim=out_dim,
                          h_dim=num_nodes, n_h_layers=num_layers)

    # print the structure of NN.
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate)
    acc_1 = 0
    for i in range(5):
        train(model, optimizer, train_loader, torch.device("cpu"))
        # Validation Error
        acc_1 = val(model, test_loader, torch.device("cpu"))
        if (i + 1) % 50 == 0:
            print('{} epoch: acc={}'.format(i + 1, acc_1))

    # Name the model!
    acc_2 = val(model, test_loader_2, torch.device('cpu'))
    name = 'ResNet_' + str(num_nodes) + '_' + str(num_layers)
    model.save(path=root + "/GP_results/{}.pth".format(name))
    print('GP iteration with acc:', acc_1, acc_2)
    acc = acc_1 * 0.3 + acc_2 * 0.7
    print('Final acc: ', acc)
    return acc


if __name__ == '__main__':
    gp_result = gp_minimize(func=loss_func,
                            dimensions=dimensions,
                            n_calls=11,
                            # noise=1e-8,
                            n_jobs=-1,
                            acq_func="EI",
                            x0=default_parameters)
    print(gp_result)
    print('Best parameters: lr={}, bs={}, n_layers={}, n_nodes={}'
          .format(gp_result.x[0], gp_result.x[1], gp_result.x[2], gp_result.x[3]))
    plot_convergence(gp_result)
