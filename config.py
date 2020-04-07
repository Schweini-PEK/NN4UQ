import warnings


class DefaultConfig(object):
    func = 'train'

    model = 'MLP'
    math_model = 'lsode'
    root = 'dataset/'
    n_data = 1500
    train_ratio = 0.9

    use_gpu = True
    print_freq = 1

    state = {'x_0': 1.0, 'delta': 0.1, 'alpha': 0.3225}


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
config = DefaultConfig()
