import numpy as np


def list2sample(n_features, dataset):
    """Get x and y(should be one value for one sample) in the ndarray format from the dataset.

    :param n_features: Number of features
    :param dataset: The original dataset
    :return: array-like of shape (n_samples, n_features) and array-like of shape (n_samples, 1)
    """
    n_samples = len(dataset)
    sample_x = np.zeros((n_samples, n_features))
    sample_y = np.zeros((n_samples, 1))

    for i in range(n_samples):
        for j in range(n_features):
            sample_x[i][j] = dataset[i][0][j]
        sample_y[i][0] = dataset[i][1]

    return sample_x, sample_y


def print_best_score(gs, param_test):
    print("Best score: %0.3f" % gs.best_score_)
    print("Best parameters set:")
    best_parameters = gs.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
