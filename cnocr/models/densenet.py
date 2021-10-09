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
# Credits: adapted from https://github.com/mindee/doctr

from typing import Tuple

from torch import Tensor
from torch import nn
from torchvision.models import densenet


class DenseNet(densenet.DenseNet):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__(
            growth_rate,
            block_config,
            num_init_features,
            bn_size,
            drop_rate,
            num_classes=1,  # useless, will be deleted
            memory_efficient=memory_efficient,
        )

        self.block_config = block_config

        delattr(self, 'classifier')
        self.features.conv0 = nn.Conv2d(
            1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.features.pool0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        last_denselayer = self._get_last_denselayer(len(self.block_config))
        conv = last_denselayer.conv2
        in_channels, out_channels = conv.in_channels, conv.out_channels
        last_denselayer.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False
        )

        # for i in range(1, len(self.block_config)):
        #     transition = getattr(self.features, 'transition%d' % i)
        #     in_channels, out_channels = transition.conv.in_channels, transition.conv.out_channels
        #     trans = _MaxPoolTransition(num_input_features=in_channels,
        #                                num_output_features=out_channels)
        #     setattr(self.features, 'transition%d' % i, trans)

        self._post_init_weights()

    def _get_last_denselayer(self, block_num):
        denseblock = getattr(self.features, 'denseblock%d' % block_num)
        i = 1
        while hasattr(denseblock, 'denselayer%d' % i):
            i += 1
        return getattr(denseblock, 'denselayer%d' % (i-1))

    @property
    def compress_ratio(self):
        return 2 ** (len(self.block_config) - 1)

    def _post_init_weights(self):
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        return features


class DenseNetLite(DenseNet):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int] = (2, 2, 2),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__(
            growth_rate,
            block_config,
            num_init_features,
            bn_size,
            drop_rate,
            memory_efficient=memory_efficient,
        )
        self.features.pool0 = nn.AvgPool2d(kernel_size=2, stride=2)

        # last max pool, pool 1/8 to 1/16 for height dimension
        self.features.add_module(
            'pool5', nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

    @property
    def compress_ratio(self):
        return 2 ** len(self.block_config)


class _MaxPoolTransition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2))
