import warnings


class DefaultConfig(object):
    model = 'MLP'
    math_model = 'mlode'
    load_model_path = "MLP_0317_22:55:02"
    root = 'dataset/'
    n_data = 1500

    use_gpu = False
    num_workers = 4
    print_freq = 1

    k_fold = 1
    batch_size = 2
    train_ratio = 0.9
    max_epoch = 2000
    lr = 0.3
    lr_decay = 0.95

    trajectory_length = 100
    x_init = 1.0
    alpha = 0.5
    delta = 0.1


def parse(self, kwargs):
    """ Update the config parameters from kwargs

    :param self:
    :param kwargs:
    :return:
    """
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribute %s" % k)
        setattr(self, k, v)

    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
options = DefaultConfig()
