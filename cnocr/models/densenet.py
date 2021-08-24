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
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
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
        self.features.conv0 = nn.Conv2d(
            1, num_init_features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.features.pool0 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        delattr(self, 'classifier')
        self._post_init_weights()

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
