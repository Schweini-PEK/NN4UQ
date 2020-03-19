import warnings


class DefaultConfig(object):
    model = 'MLP'
    # load_model_path = "checkpoints/<class 'models.shallowRes.ShallowResBN'>_0312_19:00:50.pth"
    load_model_path = "MLP_0317_22:55:02"
    root = 'dataset/'
    dataset = 'lsode'
    use_gpu = False
    print_freq = 1
    num_workers = 4

    train_size = 1500
    k_fold = 1
    batch_size = 2
    train_ratio = 0.9
    max_epoch = 200
    lr = 0.01
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
