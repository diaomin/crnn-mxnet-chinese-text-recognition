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
from glob import glob
from typing import Union, List, Tuple, Optional, Collection
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .consts import MODEL_VERSION, AVAILABLE_MODELS, VOCAB_FP
from .models.ocr_model import OcrModel
from .utils import (
    data_dir,
    get_model_file,
    read_charset,
    check_model_name,
    check_context,
    read_img,
    load_model_params,
    resize_img,
    pad_img_seq,
    to_numpy,
)
from .data_utils.aug import NormalizeAug
from .models.ctc import CTCPostProcessor

logger = logging.getLogger(__name__)


def gen_model(model_name, vocab):
    check_model_name(model_name)
    model = OcrModel.from_name(model_name, vocab)
    return model


class Recognizer(object):
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
            >>> rec = Recognizer()

            使用指定模型：
            >>> rec = Recognizer(model_name='densenet_lite_136-fc')

            识别时只考虑数字：
            >>> rec = Recognizer(model_name='densenet_lite_136-fc', cand_alphabet='0123456789')

        """
        model_backend = model_backend.lower()
        assert model_backend in ('pytorch', 'onnx')
        if 'name' in kwargs:
            logger.warning(
                'param `name` is useless and deprecated since version %s'
                % MODEL_VERSION
            )
        check_model_name(model_name)
        check_context(context)

        self._model_name = model_name
        self._model_backend = model_backend
        if context == 'gpu':
            context = 'cuda'
        self.context = context

        try:
            self._assert_and_prepare_model_files(model_fp, root)
        except NotImplementedError:
            logger.warning(
                'no available model is found for name %s and backend %s'
                % (self._model_name, self._model_backend)
            )
            self._model_backend = (
                'onnx' if self._model_backend == 'pytorch' else 'pytorch'
            )
            logger.warning(
                'trying to use name %s and backend %s'
                % (self._model_name, self._model_backend)
            )
            self._assert_and_prepare_model_files(model_fp, root)

        self._vocab, self._letter2id = read_charset(vocab_fp)
        self.postprocessor = CTCPostProcessor(vocab=self._vocab)

        self._candidates = None
        self.set_cand_alphabet(cand_alphabet)

        self._model = self._get_model(context)

    def _assert_and_prepare_model_files(self, model_fp, root):
        self._model_file_prefix = '{}-{}'.format(self.MODEL_FILE_PREFIX, self._model_name)
        model_epoch = AVAILABLE_MODELS.get_epoch(self._model_name, self._model_backend)

        if model_epoch is not None:
            self._model_file_prefix = '%s-epoch=%03d' % (
                self._model_file_prefix,
                model_epoch,
            )

        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        if model_fp is not None:
            self._model_fp = model_fp
            return

        root = os.path.join(root, MODEL_VERSION)
        self._model_dir = os.path.join(root, self._model_name)
        model_ext = 'ckpt' if self._model_backend == 'pytorch' else 'onnx'
        fps = glob('%s/%s*.%s' % (self._model_dir, self._model_file_prefix, model_ext))
        if len(fps) > 1:
            raise ValueError(
                'multiple %s files are found in %s, not sure which one should be used'
                % (model_ext, self._model_dir)
            )
        elif len(fps) < 1:
            logger.warning('no %s file is found in %s' % (model_ext, self._model_dir))
            get_model_file(
                self._model_name, self._model_backend, self._model_dir
            )  # download the .zip file and unzip
            fps = glob(
                '%s/%s*.%s' % (self._model_dir, self._model_file_prefix, model_ext)
            )

        self._model_fp = fps[0]

    def _get_model(self, context):
        logger.info('use model: %s' % self._model_fp)
        if self._model_backend == 'pytorch':
            model = gen_model(self._model_name, self._vocab)
            model.eval()
            model.to(self.context)
            model = load_model_params(model, self._model_fp, context)
        elif self._model_backend == 'onnx':
            import onnxruntime

            model = onnxruntime.InferenceSession(self._model_fp)
        else:
            raise NotImplementedError(f'{self._model_backend} is not supported yet')

        return model

    def set_cand_alphabet(self, cand_alphabet: Optional[Union[Collection, str]]):
        """
        设置待识别字符的候选集合。

        Args:
            cand_alphabet (Optional[Union[Collection, str]]): 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围

        Returns:
            None

        """
        if cand_alphabet is None:
            self._candidates = None
        else:
            cand_alphabet = [
                word if word != ' ' else '<space>' for word in cand_alphabet
            ]
            excluded = set(
                [word for word in cand_alphabet if word not in self._letter2id]
            )
            if excluded:
                logger.warning(
                    'chars in candidates are not in the vocab, ignoring them: %s'
                    % excluded
                )
            candidates = [word for word in cand_alphabet if word in self._letter2id]
            self._candidates = None if len(candidates) == 0 else candidates
            logger.debug('candidate chars: %s' % self._candidates)

    # def ocr(
    #     self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    # ) -> List[Tuple[List[str], float]]:
    #     """
    #     识别函数。
    #
    #     Args:
    #         img_fp (Union[str, Path, torch.Tensor, np.ndarray]): image file path; or color image torch.Tensor or np.ndarray,
    #             with shape [height, width] or [height, width, channel].
    #             channel should be 1 (gray image) or 3 (RGB formatted color image). scaled in [0, 255].
    #
    #     Returns:
    #         list of (list of chars, prob), such as
    #         [(['第', '一', '行'], 0.80), (['第', '二', '行'], 0.75), (['第', '三', '行'], 0.9)]
    #     """
    #     img = self._prepare_img(img_fp)
    #
    #     if min(img.shape[0], img.shape[1]) < 2:
    #         return []
    #     if img.mean() < 145:  # 把黑底白字的图片对调为白底黑字
    #         img = 255 - img
    #     line_imgs = line_split(np.squeeze(img, axis=-1), blank=True)
    #     line_img_list = [np.expand_dims(line_img, axis=-1) for line_img, _ in line_imgs]
    #     line_chars_list = self.ocr_for_single_lines(line_img_list)
    #     return line_chars_list

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
            img = read_img(img_fp)

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        elif len(img.shape) == 3:
            if img.shape[2] == 3:
                # color to gray
                img = np.expand_dims(np.array(Image.fromarray(img).convert('L')), -1)
            elif img.shape[2] != 1:
                raise ValueError(
                    'only images with shape [height, width, 1] (gray images), '
                    'or [height, width, 3] (RGB-formated color images) are supported'
                )

        if img.dtype != np.dtype('uint8'):
            img = img.astype('uint8')
        return img

    def ocr_for_single_line(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> Tuple[List[str], float]:
        """
        Recognize characters from an image with only one-line characters.

        Args:
            img_fp (Union[str, Path, torch.Tensor, np.ndarray]):
                image file path; or image torch.Tensor or np.ndarray,
                with shape [height, width] or [height, width, channel].
                The optional channel should be 1 (gray image) or 3 (color image).

        Returns:
            tuple: (list of chars, prob), such as (['你', '好'], 0.80)
        """
        img = self._prepare_img(img_fp)
        res = self.ocr_for_single_lines([img])
        return res[0]

    def recognize(
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
                The optional channel should be 1 (gray image) or 3 (RGB-format color image).
                注：img_list 不宜包含太多图片，否则同时导入这些图片会消耗很多内存。
            batch_size: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `1`。

        Returns:
            list: list of (chars, prob), such as
            [('第一行', 0.80), ('第二行', 0.75), ('第三行', 0.9)]
        """
        if len(img_list) == 0:
            return []

        img_list = [self._prepare_img(img) for img in img_list]
        img_list = [self._transform_img(img) for img in img_list]

        should_sort = batch_size > 1 and len(img_list) // batch_size > 1

        if should_sort:
            # 把图片按宽度从小到大排列，提升效率
            sorted_idx_list = sorted(
                range(len(img_list)), key=lambda i: img_list[i].shape[2]
            )
            sorted_img_list = [img_list[i] for i in sorted_idx_list]
        else:
            sorted_idx_list = range(len(img_list))
            sorted_img_list = img_list

        idx = 0
        sorted_out = []
        while idx * batch_size < len(sorted_img_list):
            imgs = sorted_img_list[idx * batch_size : (idx + 1) * batch_size]
            try:
                batch_out = self._predict(imgs)
            except Exception as e:
                # 对于太小的图片，如宽度小于8，会报错
                batch_out = {'preds': [([''], 0.0)] * len(imgs)}
            sorted_out.extend(batch_out['preds'])
            idx += 1
        out = [None] * len(sorted_out)
        for idx, pred in zip(sorted_idx_list, sorted_out):
            out[idx] = pred

        res = []
        for line in out:
            chars, prob = line
            chars = [c if c != '<space>' else ' ' for c in chars]
            res.append((''.join(chars), prob))

        return res

    def _transform_img(self, img: np.ndarray) -> torch.Tensor:
        """
        Args:
            img: image array with type torch.Tensor or np.ndarray,
            with shape [height, width] or [height, width, channel].
            channel shoule be 1 (gray image) or 3 (color image).

        Returns:
            torch.Tensor: with shape (1, height, width)
        """
        img = resize_img(img.transpose((2, 0, 1)))  # res: [C, H, W]
        return NormalizeAug()(img).to(device=torch.device(self.context))

    def _predict(self, img_list: List[torch.Tensor]):
        img_lengths = torch.tensor([img.shape[2] for img in img_list])
        imgs = pad_img_seq(img_list)
        if self._model_backend == 'pytorch':
            with torch.no_grad():
                out = self._model(
                    imgs, img_lengths, candidates=self._candidates, return_preds=True
                )
        else:  # onnx
            out = self._onnx_predict(imgs, img_lengths)

        return out

    def _onnx_predict(self, imgs, img_lengths):
        ort_session = self._model
        ort_inputs = {
            ort_session.get_inputs()[0].name: to_numpy(imgs),
            ort_session.get_inputs()[1].name: to_numpy(img_lengths),
        }
        ort_outs = ort_session.run(None, ort_inputs)
        out = {
            'logits': torch.from_numpy(ort_outs[0]),
            'output_lengths': torch.from_numpy(ort_outs[1]),
        }
        out['logits'] = OcrModel.mask_by_candidates(
            out['logits'], self._candidates, self._vocab, self._letter2id
        )

        out["preds"] = self.postprocessor(out['logits'], out['output_lengths'])
        return out
