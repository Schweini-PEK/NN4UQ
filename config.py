import logging

logger = logging.getLogger(__name__)


class DefaultConfig(object):
    def __init__(self):
        self.bs = 8
        self.model = 'MLP'
        self.lr = 0.1
        self.epoch = 5

    dp = 'dataset/george_ns.pkl'  # data path
    train_ratio = 0.9

    use_gpu = True
    sf = 100  # sample freq
    val_freq = 5
    print_freq = 50

    k = 2  # For RSResNet
    nh = 30  # The number of nodes of hidden layers.
    state = {'x_0': 1.0, 'delta': 0.1, 'alpha': 0.6}

    def parse(self, kwargs):
        """ Update the config parameters from kwargs

        :param self:
        :param kwargs:
        :return:
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                logger.warning('Opt has no attribute {}.'.format(k))
            setattr(self, k, v)


config = DefaultConfig()
