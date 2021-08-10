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
import os
import logging
from typing import Union, List, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from cnocr.consts import MODEL_VERSION, AVAILABLE_MODELS, VOCAB_FP
from cnocr.models.ocr_model import OcrModel
from cnocr.utils import (
    data_dir,
    get_model_file,
    read_charset,
    check_model_name,
    check_context,
    read_img,
    load_model_params,
    rescale_img,
    pad_img_seq,
)
from .data_utils.aug import NormalizeAug
from .line_split import line_split

logger = logging.getLogger(__name__)


def gen_model(model_name, vocab):
    check_model_name(model_name)
    if not model_name.startswith('densenet-s'):
        logger.warning(
            'only "densenet-s" is supported now, use "densenet-s-fc" by default'
        )
        model_name = 'densenet-s-fc'
    model = OcrModel.from_name(model_name, vocab)
    return model


class CnOcr(object):
    MODEL_FILE_PREFIX = 'cnocr-v{}'.format(MODEL_VERSION)

    def __init__(
        self,
        model_name='densenet-lite-fc',
        model_epoch=None,
        cand_alphabet=None,
        root=data_dir(),
        context='cpu',  # ['cpu', 'gpu', 'cuda']
        name=None,
    ):
        """

        :param model_name: 模型名称
        :param model_epoch: 模型迭代次数。默认为 None，表示使用系统自带的模型对应的迭代次数
        :param cand_alphabet: 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围
        :param root: 模型文件所在的根目录。
            Linux/Mac下默认值为 `~/.cnocr`，表示模型文件所处文件夹类似 `~/.cnocr/1.2.0/densenet-lite-fc`。
            Windows下默认值为 `C:/Users/<username>/AppData/Roaming/cnocr`。
        :param context: 'cpu', or 'gpu'。表明预测时是使用CPU还是GPU。默认为CPU。
        :param name: 正在初始化的这个实例名称。如果需要同时初始化多个实例，需要为不同的实例指定不同的名称。
        """
        check_model_name(model_name)
        check_context(context)
        self._model_name = model_name
        self._model_file_prefix = '{}-{}'.format(self.MODEL_FILE_PREFIX, model_name)
        # self._model_epoch = model_epoch or AVAILABLE_MODELS[model_name][0]

        root = os.path.join(root, MODEL_VERSION)
        self._model_dir = os.path.join(root, self._model_name)
        # self._assert_and_prepare_model_files()
        self._vocab, self._letter2id = read_charset(VOCAB_FP)

        self._candidates = None
        self.set_cand_alphabet(cand_alphabet)

        self.context = context
        self._mod = self._get_module(context)

    def _assert_and_prepare_model_files(self):
        model_dir = self._model_dir
        model_files = [
            '%s-%04d.params' % (self._model_file_prefix, self._model_epoch),
        ]
        file_prepared = True
        for f in model_files:
            f = os.path.join(model_dir, f)
            if not os.path.exists(f):
                file_prepared = False
                logger.warning('can not find file %s', f)
                break

        if file_prepared:
            return

        get_model_file(model_dir)

    def _get_module(self, context):
        from glob import glob

        fps = glob('%s/%s*.ckpt' % (self._model_dir, self._model_file_prefix))
        if len(fps) > 1:
            raise ValueError(
                'multiple ckpt files are found in %s, not sure which one should be used'
                % self._model_dir
            )
        elif len(fps) < 1:
            raise FileNotFoundError('no ckpt file is found in %s' % self._model_dir)

        fp = fps[0]
        model = gen_model(self._model_name, self._vocab)
        model.eval()
        model = load_model_params(model, fp, context)

        return model

    def set_cand_alphabet(self, cand_alphabet):
        """
        设置待识别字符的候选集合。
        :param cand_alphabet: 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围
        :return: None
        """
        if cand_alphabet is None:
            self._candidates = None
        else:
            cand_alphabet = [word if word != ' ' else '<space>' for word in cand_alphabet]
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
            logger.info('candidate chars: %s' % self._candidates)

    def ocr(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> List[Tuple[List[str], float]]:
        """
        :param img_fp: image file path; or color image mx.nd.NDArray or np.ndarray,
            with shape (height, width, 3), and the channels should be RGB formatted.
        :return: list of (list of chars, prob), such as
            [(['第', '一', '行'], 0.80), (['第', '二', '行'], 0.75), (['第', '三', '行'], 0.9)]
        """
        if isinstance(img_fp, str):
            if not os.path.isfile(img_fp):
                raise FileNotFoundError(img_fp)
            img = read_img(img_fp)
        elif isinstance(img_fp, torch.Tensor):
            img = img_fp.numpy()
        elif isinstance(img_fp, np.ndarray):
            img = img_fp
        else:
            raise TypeError('Inappropriate argument type.')
        if min(img.shape[1], img.shape[2]) < 2:
            return []
        if img.mean() < 145:  # 把黑底白字的图片对调为白底黑字
            img = 255 - img
        line_imgs = line_split(np.squeeze(img, axis=0), blank=True)
        line_img_list = [np.expand_dims(line_img, axis=0) for line_img, _ in line_imgs]
        line_chars_list = self.ocr_for_single_lines(line_img_list)
        return line_chars_list

    def ocr_for_single_line(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> Tuple[List[str], float]:
        """
        Recognize characters from an image with only one-line characters.
        :param img_fp: image file path; or image mx.nd.NDArray or np.ndarray,
            with shape [height, width] or [height, width, channel].
            The optional channel should be 1 (gray image) or 3 (color image).
        :return: (list of chars, prob), such as (['你', '好'], 0.80)
        """
        if isinstance(img_fp, str):
            if not os.path.isfile(img_fp):
                raise FileNotFoundError(img_fp)
            img = read_img(img_fp)
        elif isinstance(img_fp, torch.Tensor) or isinstance(img_fp, np.ndarray):
            img = img_fp
        else:
            raise TypeError('Inappropriate argument type.')
        res = self.ocr_for_single_lines([img])
        return res[0]

    def ocr_for_single_lines(
        self, img_list: List[Union[torch.Tensor, np.ndarray]]
    ) -> List[Tuple[List[str], float]]:
        """
        Batch recognize characters from a list of one-line-characters images.
        :param img_list: list of images, in which each element should be a line image array,
            with type mx.nd.NDArray or np.ndarray.
            Each element should be a tensor with values ranging from 0 to 255,
            and with shape [height, width] or [height, width, channel].
            The optional channel should be 1 (gray image) or 3 (color image).
        :return: list of (list of chars, prob), such as
            [(['第', '一', '行'], 0.80), (['第', '二', '行'], 0.75), (['第', '三', '行'], 0.9)]
        """
        if len(img_list) == 0:
            return []
        img_list = [self._preprocess_img_array(img) for img in img_list]

        out = self._predict(img_list)

        res = []
        for line in out['preds']:
            chars, prob = line
            chars = [c if c != '<space>' else ' ' for c in chars]
            res.append((chars, prob))

        return res

    def _preprocess_img_array(
        self, img: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        :param img: image array with type torch.Tensor or np.ndarray,
        with shape [height, width] or [channel, height, width].
        channel shoule be 1 (gray image) or 3 (color image).

        :return: torch.Tensor, with shape (1, height, width)
        """
        if len(img.shape) == 3 and img.shape[0] == 3:
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            if img.dtype != np.dtype('uint8'):
                img = img.astype('uint8')
            # color to gray
            img = np.array(Image.fromarray(img).convert('L'))
        img = rescale_img(img)
        return NormalizeAug()(img).to(device=torch.device(self.context))

    def _predict(self, img_list: List[torch.Tensor]):
        img_lengths = torch.tensor([img.shape[2] for img in img_list])
        imgs = pad_img_seq(img_list)
        with torch.no_grad():
            out = self._mod(
                imgs, img_lengths, candidates=self._candidates, return_preds=True
            )
        return out
