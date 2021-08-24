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

import random

import torch

from ..utils import normalize_img_array


class FgBgFlipAug(object):
    """前景色背景色对调。

    Parameters
    ----------
    p : float
        Probability to flip image horizontally
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, src):
        """Augmenter body"""
        if random.random() < self.p:
            src = 255 - src
        return src


class NormalizeAug(object):
    def __call__(self, img):
        return normalize_img_array(img)


class RandomPaddingAug(object):
    def __init__(self, p, max_pad_len):
        self.p = p
        self.max_pad_len = max_pad_len

    def __call__(self, img: torch.Tensor):
        """

        :param img: [C, H, W]
        :return:
        """
        if random.random() >= self.p:
            return img
        pad_len = random.randint(1, self.max_pad_len)
        pad_shape = list(img.shape)
        pad_shape[-1] = pad_len
        padding = torch.zeros(pad_shape, dtype=img.dtype, device=img.device)
        return torch.cat((img, padding), dim=-1)
