# coding: utf-8
# Copyright (C) 2021, [Breezedeus](https://github.com/breezedeus).
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
# 如: __version__ = '2.0.*'，对应的 MODEL_VERSION 都是 '2.0'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])

IMG_STANDARD_HEIGHT = 32
VOCAB_FP = Path(__file__).parent / 'label_cn.txt'

ENCODER_CONFIGS = {
    'densenet': {  # 长度压缩至 1/8（seq_len == 35），输出的向量长度为 4*128 = 512
        'growth_rate': 32,
        'block_config': [2, 2, 2, 2],
        'num_init_features': 64,
        'out_length': 512,  # 输出的向量长度为 4*128 = 512
    },
    'densenet_1112': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 1, 2],
        'num_init_features': 64,
        'out_length': 400,
    },
    'densenet_1114': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 1, 4],
        'num_init_features': 64,
        'out_length': 656,
    },
    'densenet_1122': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 2, 2],
        'num_init_features': 64,
        'out_length': 464,
    },
    'densenet_1124': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 2, 4],
        'num_init_features': 64,
        'out_length': 720,
    },
    'densenet_lite_113': {  # 长度压缩至 1/8（seq_len == 35），输出的向量长度为 2*136 = 272
        'growth_rate': 32,
        'block_config': [1, 1, 3],
        'num_init_features': 64,
        'out_length': 272,  # 输出的向量长度为 2*80 = 160
    },
    'densenet_lite_114': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 4],
        'num_init_features': 64,
        'out_length': 336,
    },
    'densenet_lite_124': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 2, 4],
        'num_init_features': 64,
        'out_length': 368,
    },
    'densenet_lite_134': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 3, 4],
        'num_init_features': 64,
        'out_length': 400,
    },
    'densenet_lite_136': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 3, 6],
        'num_init_features': 64,
        'out_length': 528,
    },
    'mobilenetv3_tiny': {'arch': 'tiny', 'out_length': 384,},
    'mobilenetv3_small': {'arch': 'small', 'out_length': 384,},
}

DECODER_CONFIGS = {
    'lstm': {'rnn_units': 128,},
    'gru': {'rnn_units': 128,},
    'fc': {'hidden_size': 128, 'dropout': 0.1,},
    'fcfull': {'hidden_size': 256, 'dropout': 0.3,},
}

root_url = (
    'https://huggingface.co/breezedeus/cnstd-cnocr-models/resolve/main/models/cnocr/%s/'
    % MODEL_VERSION
)
# name: (epoch, url)
AVAILABLE_MODELS = {
    ('densenet_lite_114-fc', 'pytorch'): (37, root_url + 'densenet_lite_114-fc.zip'),
    ('densenet_lite_124-fc', 'pytorch'): (39, root_url + 'densenet_lite_124-fc.zip'),
    ('densenet_lite_134-fc', 'pytorch'): (34, root_url + 'densenet_lite_134-fc.zip'),
    ('densenet_lite_136-fc', 'pytorch'): (39, root_url + 'densenet_lite_136-fc.zip'),
    ('densenet_lite_114-fc', 'onnx'): (37, root_url + 'densenet_lite_114-fc-onnx.zip'),
    ('densenet_lite_124-fc', 'onnx'): (39, root_url + 'densenet_lite_124-fc-onnx.zip'),
    ('densenet_lite_134-fc', 'onnx'): (34, root_url + 'densenet_lite_134-fc-onnx.zip'),
    ('densenet_lite_136-fc', 'onnx'): (39, root_url + 'densenet_lite_136-fc-onnx.zip'),
    ('densenet_lite_134-gru', 'pytorch'): (2, root_url + 'densenet_lite_134-gru.zip'),
    ('densenet_lite_136-gru', 'pytorch'): (2, root_url + 'densenet_lite_136-gru.zip'),
}

# 候选字符集合
NUMBERS = string.digits + string.punctuation
ENG_LETTERS = string.digits + string.ascii_letters + string.punctuation
