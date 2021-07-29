# coding: utf-8
import random

from ..utils import normalize_img_array


class FgBgFlipAug(object):
    """前景色背景色对调。

    Parameters
    ----------
    p : float
        Probability to flip image horizontally
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, src):
        """Augmenter body"""
        if random.random() < self.p:
            src = 255 - src
        return src


class NormalizeAug(object):
    def __call__(self, img):
        return normalize_img_array(img)
