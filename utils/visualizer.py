import time

import numpy as np
import visdom


class Visualizer(object):
    def __init__(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        self.index = {}
        self.log_text = ''
        self.init_time = time.strftime("%H:%M:%S", time.localtime())

    def re_init(self, env='main', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        :param d: dict (name,value) i.e. ('data',0.11).
        :return:
        """
        for k, v in d.iteritems():
            # self.plot(n, v)
            continue

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, win, name, y, x=None, **kwargs):
        """
        self.plot('data',1.00)
        """
        rookie = False
        if isinstance(x, type(None)):
            x = self.index.get(name, 0)
            if not x:
                rookie = True
        x = np.reshape(np.array([x]), (-1))
        y = np.reshape(np.array([y]), (-1))
        self.vis.line(Y=y, X=x,
                      win=win,
                      name=name,
                      opts=dict(title=self.init_time),
                      update=None if rookie else 'append',
                      # update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_photo',t.Tensor(64,64))
        self.img('input_photos',t.Tensor(3,64,64))
        self.img('input_photos',t.Tensor(100,1,64,64))
        self.img('input_photos',t.Tensor(100,3,64,64),n_rows=10)
        ！！！don‘t ~~self.img('input_photos',t.Tensor(100,64,64),n_rows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """self.log({'data':1,'lr':0.0001})


        :param info:
        :param win:
        :return:
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
