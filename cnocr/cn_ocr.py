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

import os
import logging
from typing import Union, List, Tuple, Optional, Collection
from pathlib import Path

import numpy as np
import torch

from .consts import MODEL_VERSION, AVAILABLE_MODELS, VOCAB_FP
from .utils import data_dir, read_img
from .line_split import line_split
from .recognizer import Recognizer
from .ppocr import PPRecognizer, PP_SPACE

logger = logging.getLogger(__name__)


class CnOcr(object):
    MODEL_FILE_PREFIX = 'cnocr-v{}'.format(MODEL_VERSION)

    def __init__(
        self,
        model_name: str = 'densenet_lite_136-fc',
        *,
        cand_alphabet: Optional[Union[Collection, str]] = None,
        context: str = 'cpu',  # ['cpu', 'gpu', 'cuda']
        model_fp: Optional[str] = None,
        model_backend: str = 'onnx',  # ['pytorch', 'onnx']
        root: Union[str, Path] = data_dir(),
        vocab_fp: Union[str, Path] = VOCAB_FP,
        **kwargs,
    ):
        """
        识别模型初始化函数。

        Args:
            model_name (str): 模型名称。默认为 `densenet_lite_136-fc`
            cand_alphabet (Optional[Union[Collection, str]]): 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围
            context (str): 'cpu', or 'gpu'。表明预测时是使用CPU还是GPU。默认为 `cpu`。
                此参数仅在 `model_backend=='pytorch'` 时有效。
            model_fp (Optional[str]): 如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（'.ckpt' 文件）
            model_backend (str): 'pytorch', or 'onnx'。表明预测时是使用 PyTorch 版本模型，还是使用 ONNX 版本模型。
                同样的模型，ONNX 版本的预测速度一般是 PyTorch 版本的2倍左右。默认为 'onnx'。
            root (Union[str, Path]): 模型文件所在的根目录。
                Linux/Mac下默认值为 `~/.cnocr`，表示模型文件所处文件夹类似 `~/.cnocr/2.1/densenet_lite_136-fc`。
                Windows下默认值为 `C:/Users/<username>/AppData/Roaming/cnocr`。
            vocab_fp (Union[str, Path]): 字符集合的文件路径，即 `label_cn.txt` 文件路径。
                若训练的自有模型更改了字符集，看通过此参数传入新的字符集文件路径。
            **kwargs: 目前未被使用。

        Examples:
            使用默认参数：
            >>> ocr = CnOcr()

            使用指定模型：
            >>> ocr = CnOcr(model_name='densenet_lite_136-fc')

            识别时只考虑数字：
            >>> ocr = CnOcr(model_name='densenet_lite_136-fc', cand_alphabet='0123456789')

        """
        self.space = AVAILABLE_MODELS.get_space(model_name, model_backend)
        if self.space == AVAILABLE_MODELS.CNOCR_SPACE:
            rec_cls = Recognizer
        elif self.space == PP_SPACE:
            rec_cls = PPRecognizer
            if vocab_fp is not None and vocab_fp != VOCAB_FP:
                logger.warning('param `vocab_fp` is invalid for %s models' % PP_SPACE)
        else:
            raise NotImplementedError(
                '%s is not supported currently' % ((model_name, model_backend),)
            )

        self.rec_model = rec_cls(
            model_name=model_name,
            cand_alphabet=cand_alphabet,
            context=context,
            model_fp=model_fp,
            root=root,
            vocab_fp=vocab_fp,
        )

    def ocr(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        识别函数。

        Args:
            img_fp (Union[str, Path, torch.Tensor, np.ndarray]): image file path; or color image torch.Tensor or np.ndarray,
                with shape [height, width] or [height, width, channel].
                channel should be 1 (gray image) or 3 (RGB formatted color image). scaled in [0, 255].

        Returns:
            list of (chars, prob), such as
            [('第一行', 0.80), ('第二行', 0.75), ('第三行', 0.9)]
        """
        img = self._prepare_img(img_fp)

        if min(img.shape[0], img.shape[1]) < 2:
            return []
        if img.mean() < 145:  # 把黑底白字的图片对调为白底黑字
            img = 255 - img
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=-1)
        line_imgs = line_split(img, blank=True)
        line_img_list = [line_img for line_img, _ in line_imgs]
        line_chars_list = self.ocr_for_single_lines(line_img_list)
        return line_chars_list

    def _prepare_img(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """

        Args:
            img_fp (Union[str, Path, torch.Tensor, np.ndarray]):
                image array with type torch.Tensor or np.ndarray,
                with shape [height, width] or [height, width, channel].
                channel should be 1 (gray image) or 3 (color image).

        Returns:
            np.ndarray: with shape (height, width, 1), dtype uint8, scale [0, 255]
        """
        img = img_fp
        if isinstance(img_fp, (str, Path)):
            if not os.path.isfile(img_fp):
                raise FileNotFoundError(img_fp)
            img = read_img(img_fp, gray=False)

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        elif len(img.shape) == 3:
            assert img.shape[2] in (1, 3)

        if img.dtype != np.dtype('uint8'):
            img = img.astype('uint8')
        return img

    def ocr_for_single_line(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> Tuple[str, float]:
        """
        Recognize characters from an image with only one-line characters.

        Args:
            img_fp (Union[str, Path, torch.Tensor, np.ndarray]):
                image file path; or image torch.Tensor or np.ndarray,
                with shape [height, width] or [height, width, channel].
                The optional channel should be 1 (gray image) or 3 (color image).

        Returns:
            tuple: (chars, prob), such as ('你好', 0.80)
        """
        img = self._prepare_img(img_fp)
        res = self.ocr_for_single_lines([img])
        return res[0]

    def ocr_for_single_lines(
        self,
        img_list: List[Union[str, Path, torch.Tensor, np.ndarray]],
        batch_size: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Batch recognize characters from a list of one-line-characters images.

        Args:
            img_list (List[Union[str, Path, torch.Tensor, np.ndarray]]):
                list of images, in which each element should be a line image array,
                with type torch.Tensor or np.ndarray.
                Each element should be a tensor with values ranging from 0 to 255,
                and with shape [height, width] or [height, width, channel].
                The optional channel should be 1 (gray image) or 3 (color image).
                注：img_list 不宜包含太多图片，否则同时导入这些图片会消耗很多内存。
            batch_size: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `1`。

        Returns:
            list: list of (chars, prob), such as
            [('第一行', 0.80), ('第二行', 0.75), ('第三行', 0.9)]
        """
        if len(img_list) == 0:
            return []

        img_list = [self._prepare_img(img) for img in img_list]
        res = self.rec_model.recognize(img_list, batch_size=batch_size)

        return res
