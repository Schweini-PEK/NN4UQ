import warnings


class DefaultConfig(object):
    model = 'ShallowResBN'
    # load_model_path = "checkpoints/<class 'models.shallowRes.ShallowResBN'>_0312_19:00:50.pth"
    load_model_path = "checkpoints/<class 'models.shallowRes.ShallowRes'>_0306_10:41:19.pth"
    root = 'dataset/'
    dataset = 'lsode'
    env = 'default'
    use_gpu = False
    print_freq = 1
    num_workers = 4

    train_size = 1500
    k_fold = 5
    batch_size = 128
    train_ratio = 0.9
    max_epoch = 400
    lr = 0.01
    lr_decay = 0.95

    trajectory_length = 2
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
