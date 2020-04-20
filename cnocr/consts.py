# coding: utf-8
import os
import string
from .__version__ import __version__


EMB_MODEL_TYPES = ['conv', 'conv-lite', 'densenet', 'densenet-lite']
SEQ_MODEL_TYPES = ['lstm', 'gru', 'fc']

root_url = (
    'https://raw.githubusercontent.com/breezedeus/cnocr-models/master/models/%s'
    % __version__
)
# name: (epochs, url)
AVAILABLE_MODELS = {
    'conv-lstm': (50, root_url + '/conv-lstm-0050.zip'),
    'conv-lite-lstm': (45, root_url + '/conv-lite-lstm-0045.zip'),
    'conv-lite-fc': (27, root_url + '/conv-lite-fc-0027.zip'),
    'densenet-lite-lstm': (42, root_url + '/densenet-lite-lstm-0042.zip'),
    'densenet-lite-fc': (32, root_url + '/densenet-lite-fc.zip'),
}

# 候选字符集合
NUMBERS = string.digits + string.punctuation
ENG_LETTERS = string.digits + string.ascii_letters + string.punctuation
