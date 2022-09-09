# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).
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
from typing import Tuple

import torch
import torchvision.transforms.functional as F
try:
    from torchvision.transforms.functional import get_image_size
except:
    from torchvision.transforms.functional import _get_image_size as get_image_size

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


class RandomStretchAug(object):
    """对图片在宽度上做随机拉伸"""

    def __init__(self, min_ratio=0.9, max_ratio=1.1):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img: torch.Tensor):
        """

        :param img: [C, H, W]
        :return:
        """
        _, h, w = img.shape
        new_w_ratio = self.min_ratio + random.random() * (
            self.max_ratio - self.min_ratio
        )
        return F.resize(img, [h, int(w * new_w_ratio)])


class RandomCrop(torch.nn.Module):
    def __init__(
        self, crop_size: Tuple[int, int], interpolation=F.InterpolationMode.BILINEAR
    ):
        super().__init__()
        self.crop_size = crop_size
        self.interpolation = interpolation

    def get_params(self, ori_w, ori_h) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        while True:
            h_top, h_bot = (
                random.randint(0, self.crop_size[0]),
                random.randint(0, self.crop_size[0]),
            )
            w_left, w_right = (
                random.randint(0, self.crop_size[1]),
                random.randint(0, self.crop_size[1]),
            )
            h = ori_h - h_top - h_bot
            w = ori_w - w_left - w_right
            if h < ori_h * 0.5 or w < ori_w * 0.5:
                continue

            return h_top, w_left, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        ori_w, ori_h = get_image_size(img)
        i, j, h, w = self.get_params(ori_w, ori_h)
        return F.resized_crop(img, i, j, h, w, (ori_h, ori_w), self.interpolation)


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
