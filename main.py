import getopt
import pickle
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


class Swish(nn.Module):
    """Custom the Swish activation function.

    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


class BatchNet(nn.Module):
    """An implementation of the neural network, with batch normalization and ResNet.

    """

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(BatchNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.Tanh())
        self.drop_layer1 = nn.Dropout(p=0.3)
        # The activation function of the 2nd layer is replaced by Swish.
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), Swish())
        self.drop_layer2 = nn.Dropout(p=0.3)
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = self.drop_layer1(x1)
        x2 = self.layer2(x1)
        x2 = self.drop_layer1(x2)
        x3 = self.layer3(x2)
        return x3 + torch.unsqueeze(x[:, 0], 1)


class ODEDataset(Dataset):
    """A class to create dataset after pickle loading.

    """

    def __init__(self, path):
        with open(path, 'rb') as f:
            self.ode_frame = pickle.load(f)

    def __getitem__(self, item):
        x, y = self.ode_frame[item]
        x = torch.from_numpy(np.array(x))
        y = torch.from_numpy(np.array(y))

        return x, y

    def __len__(self):
        return len(self.ode_frame)


def main(argv):
    path = 'TrainingData.pkl'
    learning_rate = 0.2
    batch = 8
    epoch_number = 500

    try:
        opts, args = getopt.getopt(argv[1:], "hi:", ['learning_rate=', 'batch_size=', 'epoch='])
    except getopt.GetoptError:
        print('data_generating.py -n <input path> --learning_rate <learning rate> --batch_size <batch size>'
              '--epoch <epoch number>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('data_generating.py -n <input path> --learning_rate <learning rate> --batch_size <batch size>'
                  '--epoch <epoch number>')
            sys.exit()
        elif opt == '-i':
            path = arg
        elif opt == '--learning_rate':
            learning_rate = arg
        elif opt == '--batch_size':
            batch = arg
        elif opt == '--epoch':
            epoch_number = arg

    model = BatchNet(3, 40, 40, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()

    ode_dataset = ODEDataset(path)
    data_loader = DataLoader(ode_dataset, batch_size=batch, shuffle=True, num_workers=4)
    for data in data_loader:
        x, y = data
        out = model(x.float())
        loss = criterion(out, y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_number += 1  # Maybe add batch size
        if epoch_number % 50 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch_number, loss.data.item()))


if __name__ == '__main__':
    main(sys.argv)

# TO DO LIST

"""
5. When the net bugs are fixed, train and get validation & test error plots (w / n of epochs)
6. Add a dropout layer (prob ~0.3) and see what happens with errors
"""
