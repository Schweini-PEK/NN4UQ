import warnings


class DefaultConfig(object):
    model = 'ShallowRes'
    load_model_path = "checkpoints/<class 'models.shallowRes.ShallowRes'>_0306_10:41:19.pth"
    load_testset_path = 'testset.pkl'
    root = 'dataset/'
    dataset = 'lsode'
    env = 'default'
    use_gpu = False

    k_fold = 5
    batch_size = 128
    train_ratio = 0.9
    num_workers = 4
    print_freq = 50

    result_file = 'result.csv'

    max_epoch = 400
    lr = 0.01
    lr_decay = 0.95

    trajectory_length = 100


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
