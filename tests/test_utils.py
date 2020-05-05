import os
import sys
import pytest
import mxnet as mx
from mxnet.gluon.utils import download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr.utils import check_context


@pytest.mark.skip()
def test_download():
    url = 'https://www.dropbox.com/s/5n09nxf4x95jprk/cnocr-models-v0.1.0.zip?dl=1'
    download(url, './cnocr-models.zip', overwrite=True)


@pytest.mark.parametrize('context, expected', [
    ('gpu', True),
    ('cpu', True),
    ('', False),
    ('xx', False),
    (mx.cpu(), True),
    (mx.gpu(), True),
    ([mx.cpu()], True),
    ([mx.gpu()], True),
    ([mx.gpu(0), mx.gpu(1)], True),
    ([], False),
])
def test_check_context(context, expected):
    assert check_context(context) == expected