import numpy as np

import utils


def data_generating(generate_func, n_data):
    return eval(generate_func)(n_data)


def lsode_generating(train_size):
    """Generate the whole training set.

    :param train_size: The number of samples.
    :return: A idx like [[x, alpha, delta], y]
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
        x_next = utils.ode.ode_predictor(x_cur, alpha, delta)
        dataset.append([[x_cur, alpha, delta], x_next])

    return dataset


def mlode_generating(train_size):
    dataset = []
    d_lower_bound = 0.0
    d_upper_bound = 2.0
    delta = 0.1
    a = np.array([[1, -4], [4, -7]])

    for i in range(train_size):
        x = np.random.uniform(d_lower_bound, d_upper_bound, 2)
        x_next = utils.ode.multi_ode_predictor(x, a, delta)
        dataset.append([[x], [x_next]])

    return dataset
