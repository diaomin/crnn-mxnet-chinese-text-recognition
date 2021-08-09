# coding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import string
from pathlib import Path
from .__version__ import __version__


# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '1.3.*'，对应的 MODEL_VERSION 都是 '1.3.0'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2]) + '.0'

IMG_STANDARD_HEIGHT = 32
VOCAB_FP = Path(__file__).parent.parent / 'label_cn.txt'

ENCODER_CONFIGS = {
    'densenet-s': {  # 长度压缩至 1/8（seq_len == 35），输出的向量长度为 4*128 = 512
        'growth_rate': 32,
        'block_config': [2, 2, 2, 2],
        'num_init_features': 64,
        'out_length': 512,  # 输出的向量长度为 4*128 = 512
    },
}

DECODER_CONFIGS = {
    'lstm': {
        'input_size': 512,  # 对应 encoder 的输出向量长度
        'rnn_units': 128,
    },
    'gru': {
        'input_size': 512,  # 对应 encoder 的输出向量长度
        'rnn_units': 128,
    },
    'fc': {
        'input_size': 512,  # 对应 encoder 的输出向量长度
        'hidden_size': 256,
        'dropout': 0.3,
    }
}

# EMB_MODEL_TYPES = [
#     'conv',  # seq_len == 35, deprecated
#     'conv-lite',  # seq_len == 69
#     'conv-lite-s',  # seq_len == 35
#     'densenet',  # seq_len == 70, deprecated
#     'densenet-lite',  # seq_len == 70
#     'densenet-s',  # seq_len == 35
#     'densenet-lite-s',  # seq_len == 35
# ]
# SEQ_MODEL_TYPES = ['lstm', 'gru', 'fc']

root_url = (
    'https://static.einplus.cn/cnocr/%s'
    % MODEL_VERSION
)
# name: (epochs, url)
AVAILABLE_MODELS = {
    'conv-lite-fc': (25, root_url + '/conv-lite-fc.zip'),
    'densenet-lite-gru': (39, root_url + '/densenet-lite-gru.zip'),
    'densenet-lite-fc': (40, root_url + '/densenet-lite-fc.zip'),
    'densenet-lite-s-gru': (35, root_url + '/densenet-lite-s-gru.zip'),
    'densenet-lite-s-fc': (40, root_url + '/densenet-lite-s-fc.zip'),
}

# 候选字符集合
NUMBERS = string.digits + string.punctuation
ENG_LETTERS = string.digits + string.ascii_letters + string.punctuation
