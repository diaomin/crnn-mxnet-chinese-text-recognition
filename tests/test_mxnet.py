# coding: utf-8
import os
import sys
import mxnet as mx
import numpy as np
from mxnet import nd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))


def test_nd():
    ele = np.reshape(np.array(range(2*3)), (2, 3))
    data = [ele, ele + 10]
    new = nd.array([ele])
    assert new.shape == (1, 2, 3)
    new = nd.array(data)
    assert new.shape == (2, 2, 3)
    print(new)
