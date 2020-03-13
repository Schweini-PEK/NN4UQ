import getopt
import pickle
import sys

import numpy as np

import utils


def train_data_generating(n_samples):
    """Generate the whole training set.

    :param n_samples: The number of samples.
    :return: A list like [[x, alpha, delta], y]
    """

    dataset = []
    p_lower_bound = 0.0
    p_upper_bound = 1.0
    s_lower_bound = 0.1
    s_upper_bound = 0.8
    delta = 0.1

    for i in range(n_samples):
        alpha = np.random.uniform(p_lower_bound, p_upper_bound)
        x0 = np.random.uniform(s_lower_bound, s_upper_bound)
        x_next = utils.lsode.state_predictor(x0, alpha, delta)
        dataset.append([[x0, alpha, delta], x_next])

    return dataset


def main(argv):
    number_train = 10
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
            number_train = arg
        elif opt == '-o':
            root = arg
        elif opt == '--model':
            sys_model = arg

    training_data = train_data_generating(number_train)
    path = root + sys_model + str(number_train) + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(training_data, f)


if __name__ == '__main__':
    main(sys.argv)
