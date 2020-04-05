import warnings


class DefaultConfig(object):
    func = 'train'

    model = 'MLP'
    math_model = 'lsode'
    load_model_path = "MLP_0322_19:02:47"
    load_result_path = ''
    root = 'dataset/'
    n_data = 2000

    use_gpu = True
    colab = True
    num_workers = 4
    print_freq = 1

    k_fold = 1
    batch_size = 2
    train_ratio = 0.9
    max_epoch = 1
    lr = 0.3
    lr_decay = 0.95

    trajectory_length = 300
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
