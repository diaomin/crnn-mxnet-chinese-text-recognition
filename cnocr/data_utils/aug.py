import random
import numpy as np
from PIL import Image
from mxnet import nd
from mxnet.image import Augmenter


class GrayAug(Augmenter):
    """FIXME don't use this one"""
    def __call__(self, img):
        """

        :param img: nd.NDArray with shape [height, width, channel] and dtype 'float32'. channel should be 3.
        :return: nd.NDArray with shape [height, width, 1] and dtype 'uint8'.
        """
        if img.dtype != np.uint8:
            img = img.astype('uint8')
        # color to gray
        img = np.array(Image.fromarray(img.asnumpy()).convert('L'))
        return nd.expand_dims(nd.array(img, dtype='uint8'), 2)


class FgBgFlipAug(Augmenter):
    """前景色背景色对调。

    Parameters
    ----------
    p : float
        Probability to flip image horizontally
    """
    def __init__(self, p):
        super(FgBgFlipAug, self).__init__(p=p)
        self.p = p

    def __call__(self, src):
        """Augmenter body"""
        if random.random() < self.p:
            src = 255 - src
        return src
