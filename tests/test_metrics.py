# coding: utf-8
import os
import sys
import mxnet as mx
from mxnet import nd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr.fit.ctc_metrics import CtcMetrics


@pytest.mark.parametrize('input, expected', [
    ('1100220030', '123'),
    ('111010220030', '1123'),
    ('121000220030', '12123'),
    ('12100022003', '12123'),
    ('012100022003', '12123'),
    ('0121000220030', '12123'),
    ('0000', ''),
])
def test_ctc_metrics(input, expected):
    input = list(map(int, list(input)))
    expected = list(map(int, list(expected)))
    p, _ = CtcMetrics.ctc_label(input)
    assert expected == p
