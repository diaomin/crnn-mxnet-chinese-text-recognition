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
# adapted from: torchvision/models/mobilenetv3.py

from functools import partial
from typing import Any, List, Optional, Callable

from torch import nn, Tensor
try:
    from torchvision.models.mobilenetv2 import ConvBNActivation
except:
    # FIXME: 目前的识别模型其实没有用到mobilenetv3的，所以这个文件应该不会被真的用到
    #   如果真用到，需要check一下新的Conv2dNormActivation是否能替代之前的ConvBNActivation
    #   Ref: https://github.com/pytorch/vision/releases/tag/v0.14.0
    from torchvision.ops.misc import Conv2dNormActivation as ConvBNActivation
from torchvision.models import mobilenetv3
from torchvision.models.mobilenetv3 import InvertedResidualConfig


class MobileNetV3(mobilenetv3.MobileNetV3):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(inverted_residual_setting, 1, 2, block, norm_layer)
        delattr(self, 'classifier')

        firstconv_input_channels = self.features[0][0].out_channels
        self.features[0] = ConvBNActivation(
            1,
            firstconv_input_channels,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=nn.Hardswish,
        )

        lastconv_input_channels = self.features[-1][0].in_channels
        lastconv_output_channels = 2 * lastconv_input_channels
        self.features[-1] = ConvBNActivation(
            lastconv_input_channels,
            lastconv_output_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.Hardswish,
        )
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        self._post_init_weights()

    @property
    def compress_ratio(self):
        return 8

    def _post_init_weights(self):
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        features = self.avgpool(features)
        return features


def _mobilenet_v3_conf(
    arch: str,
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    dilated: bool = False,
    **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(
        InvertedResidualConfig.adjust_channels, width_mult=width_mult
    )

    if arch == "mobilenet_v3_tiny":
        inverted_residual_setting = [
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, False, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 120, 48, False, "HS", 1, 1),
            # bneck_conf(48, 5, 144, 48, False, "HS", 1, 1),
            bneck_conf(
                48, 5, 288, 96 // reduce_divider, False, "HS", 2, dilation
            ),  # C4
            bneck_conf(
                96 // reduce_divider,
                5,
                128 // reduce_divider,
                96 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
            bneck_conf(
                96 // reduce_divider,
                5,
                128 // reduce_divider,
                96 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
        ]
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 1, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, False, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, False, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, False, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(
                96 // reduce_divider,
                5,
                576 // reduce_divider,
                96 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
            bneck_conf(
                96 // reduce_divider,
                5,
                576 // reduce_divider,
                96 // reduce_divider,
                True,
                "HS",
                1,
                dilation,
            ),
        ]
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting


def _mobilenet_v3_model(
    inverted_residual_setting: List[InvertedResidualConfig], **kwargs: Any
):
    model = MobileNetV3(inverted_residual_setting, **kwargs)
    return model


def gen_mobilenet_v3(arch: str = 'tiny', **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        arch (str): arch name; values: 'tiny' or 'small'

    """
    arch = 'mobilenet_v3_%s' % arch
    inverted_residual_setting = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3_model(inverted_residual_setting, **kwargs)
