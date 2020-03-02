import getopt
import pickle
import sys

import numpy as np
import scipy.integrate


def ode_model(t, x, alpha):
    """An implementation of ODE model.

    :param t: The current timestamp.
    :param x: The current state variable vector, 1 * n.
    :param alpha: The random parameter.
    :return: a n * 1 vector.
    """
    dxdt = np.zeros((1, len(x)))
    dxdt[0, 0] = -alpha * x[0]
    return [np.transpose(dxdt)]


def state_predictor(x0, alpha, dj):
    """Predict the state after dj as the length of the simulation time of the ODE model.

    :param x0: The initial state variable.
    :param alpha: The random parameter.
    :param dj: The time lag between two experiment measurements.
    :return: ???
    """
    tspan = [0, dj]
    x0 = [x0]
    alpha = alpha
    ode_sol = scipy.integrate.solve_ivp(lambda t, x: ode_model(t, x, alpha), tspan, x0, method='LSODA')
    # LSODA is the most consistent solver
    xs = ode_sol.y
    return xs.transpose()[-1, 0]


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
        # np.random.random() might be better? In this case alpha is uniformly distributed.
        # I'm not sure which distribution works better for you.
        alpha = np.random.uniform(p_lower_bound, p_upper_bound)
        x0 = np.random.uniform(s_lower_bound, s_upper_bound)
        x_next = state_predictor(x0, alpha, delta)
        dataset.append([[x0, alpha, delta], x_next])

    return dataset


def main(argv):
    number_train = 100
    path = 'TrainingData.pkl'

    try:
        opts, args = getopt.getopt(argv[1:], "hn:o:")
    except getopt.GetoptError:
        print('data_generating.py -n <the number of samples> -o <output path>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('data_generating.py -n <the number of samples> -o <output path>')
            sys.exit()
        elif opt == '-n':
            number_train = arg
        elif opt == '-o':
            path = arg

    training_data = train_data_generating(number_train)
    with open(path, 'wb') as f:
        pickle.dump(training_data, f)


if __name__ == '__main__':
    main(sys.argv)
