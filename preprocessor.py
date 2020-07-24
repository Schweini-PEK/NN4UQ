import argparse
import csv
import pickle
from os import listdir
from os.path import join

import torch


def preprocessor(src, trg, name, test=False):
    """Load and save training/test data from dir.

    Get all the csv files from a dir. Typically the names of the files are their alphas.
    :param src: The dir of data.
    :param trg: The dir for saving.
    :param name: Rename the saved file if needed.
    :param test: True for loading test set.
    :return: None
    """
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
                    sample.append(list(map(float, line[1:])))
                samples.append([alpha, sample])
                n_x = len(sample[0])
                print(n_x)
                print(sample[0])

            else:
                x_prior = list(map(float, next(reader)[1:]))
                n_x = len(x_prior)
                for line in reader:
                    x_future = list(map(float, line[1:]))
                    samples.append([torch.tensor(x_prior + alpha),
                                    torch.tensor(x_future)])
                    x_prior = x_future

    if name is None:
        if test:
            name = trg + 'NS_truth_x{}a{}.pkl'.format(n_x, n_a)

        else:
            name = trg + 'data_{}_x{}a{}.pkl'.format(len(samples), n_x, n_a)
    else:
        name = trg + name + '.pkl'
    with open(name, 'wb') as f_save:
        pickle.dump(samples, f_save)
    print('Save data at {}'.format(name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str, help="Input dir")
    parser.add_argument("--output", "-o", default="dataset/", type=str, help="Output dir")
    parser.add_argument("--test", "-t", default=False, type=bool, help="Test set or not")
    parser.add_argument("--name", "-n", default=None, help="File name")
    args = parser.parse_args()

    preprocessor(args.input, args.output, args.name, args.test)
