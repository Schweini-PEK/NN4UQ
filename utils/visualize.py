import time

import numpy as np
import visdom


class Visualizer(object):
    """

    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def re_init(self, env='default', **kwargs):
        """

        :param env:
        :param kwargs:
        :return:
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def plot(self, name, y, **kwargs):
        """self.plot('loss', 1.00)

        :param name:
        :param y:
        :param kwargs:
        :return:
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append', **kwargs)
        self.index[name] = x + 1

    def log(self, info, win='log_text'):
        """self.log({'loss':1,'lr':0.0001})

        :param self:
        :param info:
        :param win:
        :return:
        """

        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)
