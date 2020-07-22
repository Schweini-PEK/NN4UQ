import csv
import pickle
from os import listdir
from os.path import join

import torch


def preprocessor(src, trg, test=False):
    samples = []
    n_a, n_x = 0, 0
    for f_path in listdir(src):
        alpha = list(map(float, f_path[:-4].split('_')[2::2]))
        n_a = len(alpha)

        with open(join(src, f_path), 'r') as f:
            reader = csv.reader(f)

            if test:
                sample = []
                for line in reader:
                    x_future = list(map(float, line[1:]))
                    sample.append(x_future)
                samples.append([alpha, sample])
                n_x = len(sample[0])

            else:
                x_prior = list(map(float, next(reader)[1:]))
                n_x = len(x_prior)
                for line in reader:
                    x_future = list(map(float, line[1:]))
                    samples.append([torch.tensor(x_prior + alpha),
                                    torch.tensor(x_future)])
                    x_prior = x_future

    path = None
    if test:
        path = trg + '/NS_truth_x{}a{}.pkl'.format(n_x, n_a)

    else:
        path = trg + '/data_{}_x{}a{}.pkl'.format(len(samples), n_x, n_a)
    with open(path, 'wb') as f_save:
        pickle.dump(samples, f_save)
    print('Save data at {}'.format(path))


if __name__ == '__main__':
    source = "/Users/schweini/Downloads/Data_Non_Smooth"
    # source = "/Users/schweini/Desktop/test"
    target = 'dataset'

    preprocessor(source, target, test=False)
