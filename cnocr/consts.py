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
    'densenet-1112': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 1, 2],
        'num_init_features': 64,
        'out_length': 400,
    },
    'densenet-1114': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 1, 4],
        'num_init_features': 64,
        'out_length': 656,
    },
    'densenet-1122': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 2, 2],
        'num_init_features': 64,
        'out_length': 464,
    },
    'densenet-1124': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 2, 4],
        'num_init_features': 64,
        'out_length': 720,
    },

    'densenet-lite-113': {  # 长度压缩至 1/8（seq_len == 35），输出的向量长度为 2*136 = 272
        'growth_rate': 32,
        'block_config': [1, 1, 3],
        'num_init_features': 64,
        'out_length': 272,  # 输出的向量长度为 2*80 = 160
    },
    'densenet-lite-114': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 4],
        'num_init_features': 64,
        'out_length': 336,
    },
    'densenet-lite-124': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 2, 4],
        'num_init_features': 64,
        'out_length': 368,
    },
    'densenet-lite-134': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 3, 4],
        'num_init_features': 64,
        'out_length': 400,
    },
    'densenet-lite-136': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 3, 6],
        'num_init_features': 64,
        'out_length': 528,
    },
}

DECODER_CONFIGS = {
    'lstm': {
        # 'input_size': 512,  # 对应 encoder 的输出向量长度
        'rnn_units': 128,
    },
    'gru': {
        # 'input_size': 512,  # 对应 encoder 的输出向量长度
        'rnn_units': 128,
    },
    'fc': {
        # 'input_size': 512,  # 对应 encoder 的输出向量长度
        'hidden_size': 256,
        'dropout': 0.3,
    },
    'fclite': {
        # 'input_size': 512,  # 对应 encoder 的输出向量长度
        'hidden_size': 128,
        'dropout': 0.1,
    },
}

root_url = (
    'https://beiye-model.oss-cn-beijing.aliyuncs.com/models/cnocr/%s/'
    % MODEL_VERSION
)
# name: (epochs, url)
AVAILABLE_MODELS = {
    'densenet-s-fc': (8, root_url + 'densenet-s-fc-v2.0.1.zip'),
    'densenet-s-gru': (14, root_url + 'densenet-s-gru-v2.0.1.zip'),
    # 'densenet-lite-113-fclite': (33, root_url + '.zip'),
    'densenet-lite-114-fclite': (31, root_url + '.zip'),
    'densenet-lite-124-fclite': (36, root_url + '.zip'),
    'densenet-lite-134-fclite': (38, root_url + '.zip'),
    'densenet-lite-136-fclite': (38, root_url + '.zip'),
}

# 候选字符集合
NUMBERS = string.digits + string.punctuation
ENG_LETTERS = string.digits + string.ascii_letters + string.punctuation
