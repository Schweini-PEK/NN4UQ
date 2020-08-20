import argparse
import csv
import logging
import pickle
import random
from os import listdir
from os.path import join

import scipy.io
import torch

logger = logging.getLogger(__name__)


def processor(src, trg, name, type, test=False):
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
    if type == 'csv':
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

                else:
                    x_past = list(map(float, next(reader)[1:]))
                    n_x = len(x_past)
                    for line in reader:
                        try:
                            x_future = list(map(float, line[1:]))
                            samples.append([torch.tensor(x_past + alpha),
                                            torch.tensor(x_future)])
                            x_past = x_future
                        except ValueError as e:
                            print('In file {}, check the line: {}'.format(f_path, line))
                            raise

    elif type == 'mat':
        if not test:
            data = scipy.io.loadmat(src)
            x_raw = data['Xdata']
            y_raw = data['Ydata']
            index = list(range(len(x_raw)))
            random.shuffle(index)
            for i in index[:1000]:
                samples.append([torch.tensor(x_raw[i]), torch.tensor(y_raw[i])])

            n_x = len(y_raw[0])
            n_a = len(x_raw[0]) - n_x

    if name is None:
        name = src.split('/')[-1].split('.')[0] + '_x{}a{}'.format(n_x, n_a)
        if test:
            name = 'test_' + name
        else:
            name += '_{}'.format(len(samples))

    pth = trg + name + '.pkl'
    with open(pth, 'wb') as f_save:
        pickle.dump(samples, f_save)
    print('Save {} samples at {}'.format(len(samples), pth))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str, help="Input dir")
    parser.add_argument("--output", "-o", default="dataset/", type=str, help="Output dir")
    parser.add_argument("--test", default=False, type=bool, help="Test set or not")
    parser.add_argument("--name", "-n", default=None, help="File name")
    parser.add_argument("--type", "-t", default='csv', help='Processor type')
    args = parser.parse_args()

    if args.type in {'csv', 'mat'}:
        processor(args.input, args.output, args.name, args.type, args.test)
    else:
        raise NameError('Wrong file type: {}'.format(args.type))
