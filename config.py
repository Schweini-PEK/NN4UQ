import logging

logger = logging.getLogger(__name__)


class DefaultConfig(object):
    def __init__(self):
        self.bs = 8
        self.model = 'MLP'
        self.lr = 0.1
        self.epoch = 5

    dp = 'dataset/george_ns.pkl'  # data path
    ratio = 0.9

    use_gpu = True
    max_concurrent = 6
    n_samples = 2
    sf = 1  # sample freq
    val_freq = 5
    print_freq = 50

    k = 3  # For RSResNet
    nh = 30  # The number of nodes of hidden layers.
    state = {'length': 300, 'alpha': 0.3225, 'x': 1.0}

    # state = [{'x_0': [0],
    #          'alpha': [14.16, 1.0104, 0.19249, 0.83487, 0.028437], 'length': 200},
    #          {'x_0': [0],
    #           'alpha': [12.702, 3.9022, 0.25695, 0.84427, 0.011431], 'length': 200},
    #          ]
    #

    def parse(self, kwargs):
        """ Update the grid parameters from kwargs

        :param self:
        :param kwargs:
        :return:
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                logger.warning('Opt has no attribute {}.'.format(k))
            setattr(self, k, v)


config = DefaultConfig()
