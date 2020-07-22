import logging

logger = logging.getLogger(__name__)


class DefaultConfig(object):
    data_path = '/Users/schweini/Desktop/CUBES/Codes/NN4UQ/dataset/NS_24000_x3a5.pkl'
    # data_path = '/global/home/users/schweini/research/NN4UQ/dataset/data_18000_x3a4.pkl'
    ratio = 0.9

    model = 'RSResNet'
    use_gpu = True
    max_concurrent = 6
    n_samples = 2
    sf = 1  # sample freq
    val_freq = 5
    log_interval = 50

    # Default parameters for all the neural nets
    h_dim = (4, 15)  # The dimension of hidden layers
    n_h_layers = (2, 6)  # The number of hidden layers
    bs = (2, 20)
    lr = (1e-4, 1e-2)
    epoch = (100, 300)

    # Default parameters for RSResNet and RTResNet
    k = 3

    # Default parameters for testing
    truth_path = 'dataset/truth_x3a4.pkl'

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
