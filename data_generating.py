import getopt
import pickle
import sys

import numpy as np

import utils
from config import options


def train_data_generating(train_size):
    """Generate the whole training set.

    :param train_size: The number of samples.
    :return: A list like [[x, alpha, delta], y]
    """

    dataset = []
    p_lower_bound = 0.0
    p_upper_bound = 1.0
    s_lower_bound = 0.1
    s_upper_bound = 0.8
    delta = 0.1

    for i in range(train_size):
        alpha = np.random.uniform(p_lower_bound, p_upper_bound)
        x0 = np.random.uniform(s_lower_bound, s_upper_bound)
        x_cur = x0
        for j in range(1):
            x_next = utils.lsode.state_predictor(x_cur, alpha, delta)
            dataset.append([[x_cur, alpha, delta], x_next])
            x_cur = x_next

    return dataset


def main(argv):
    train_size = options.train_size
    root = 'dataset/'
    sys_model = "LSODE"

    try:
        opts, args = getopt.getopt(argv[1:], "hn:o:", ['model='])
    except getopt.GetoptError:
        print('data_generating.py -n <the number of samples> -o <output path> --model <system models>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('data_generating.py -n <the number of samples> -o <output path> --model <system models>')
            sys.exit()
        elif opt == '-n':
            train_size = arg
        elif opt == '-o':
            root = arg
        elif opt == '--model':
            sys_model = arg

    training_data = train_data_generating(train_size)
    path = root + sys_model + str(train_size) + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(training_data, f)


if __name__ == '__main__':
    main(sys.argv)
