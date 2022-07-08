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

import os
import sys
import logging
import time
from pprint import pformat

import pytest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr import CnOcr
from cnocr.utils import set_logger, read_img
from cnocr.consts import NUMBERS, AVAILABLE_MODELS
from cnocr.line_split import line_split

logger = set_logger(log_level=logging.INFO)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
example_dir = os.path.join(root_dir, 'docs/examples')
CNOCR = CnOcr(rec_model_name='densenet_lite_136-fc', det_model_name='naive_det')

SINGLE_LINE_CASES = [
    ('20457890_2399557098.jpg', ['就会哈哈大笑。3.0']),
    ('rand_cn1.png', ['笠淡嘿骅谧鼎皋姚歼蠢驼耳胬挝涯狗蒽孓犷']),
    ('rand_cn2.png', ['凉芦']),
    ('helloworld.jpg', ['Hello world!你好世界']),
]
MULTIPLE_LINE_CASES = [
    ('hybrid.png', ['o12345678']),
    (
        'multi-line_en_black.png',
        [
            'transforms the image many times. First, the image goes through many convolutional layers. In those',
            'convolutional layers, the network learns new and increasingly complex features in its layers. Then the ',
            'transformed image information goes through the fully connected layers and turns into a classification ',
            'or prediction.',
        ],
    ),
    (
        'multi-line_en_white.png',
        [
            'This chapter is currently only available in this web version. ebook and print will follow.',
            'Convolutional neural networks learn abstract features and concepts from raw image pixels. Feature',
            'Visualization visualizes the learned features by activation maximization. Network Dissection labels',
            'neural network units (e.g. channels) with human concepts.',
        ],
    ),
    (
        'multi-line_cn1.png',
        [
            '网络支付并无本质的区别，因为',
            '每一个手机号码和邮件地址背后',
            '都会对应着一个账户--这个账',
            '户可以是信用卡账户、借记卡账',
            '户，也包括邮局汇款、手机代',
            '收、电话代收、预付费卡和点卡',
            '等多种形式。',
        ],
    ),
    (
        'multi-line_cn2.png',
        [
            '当然，在媒介越来越多的情形下,',
            '意味着传播方式的变化。过去主流',
            '的是大众传播,现在互动性和定制',
            '性带来了新的挑战——如何让品牌',
            '与消费者更加互动。',
        ],
    ),
]
CASES = SINGLE_LINE_CASES + MULTIPLE_LINE_CASES


def print_preds(pred):
    print("Predicted Chars:\n", pformat(pred))


def cal_score(preds, expected):
    import Levenshtein

    if len(preds) != len(expected):
        return 0
    total_cnt = 0
    total_dist = 0
    for real, _pred in zip(expected, preds):
        pred = _pred['text']
        distance = Levenshtein.distance(real, pred)
        total_dist += distance
        total_cnt += len(real)

    return 1.0 - float(total_dist) / total_cnt


@pytest.mark.parametrize('img_fp, expected', CASES)
def test_ocr(img_fp, expected):
    ocr = CNOCR
    img_fp = os.path.join(example_dir, img_fp)

    pred = ocr.ocr(img_fp)
    print('\n')
    print_preds(pred)
    assert cal_score(pred, expected) >= 0.8

    img = read_img(img_fp)
    pred = ocr.ocr(img)
    print_preds(pred)
    assert cal_score(pred, expected) >= 0.8

    img = read_img(img_fp, gray=False)
    pred = ocr.ocr(img)
    print_preds(pred)
    assert cal_score(pred, expected) >= 0.8


@pytest.mark.parametrize('img_fp, expected', CASES)
def test_all_models(img_fp, expected):
    """测试各种模型是否可正常调用。"""
    model_name_backend_sets = AVAILABLE_MODELS.all_models()
    for model_name, model_backend in model_name_backend_sets:
        print(f'{model_name=}, {model_backend}')
        ocr = CnOcr(
            rec_model_name=model_name,
            rec_model_backend=model_backend,
            det_model_name='naive_det',
        )
        img_fp = os.path.join(example_dir, img_fp)

        pred = ocr.ocr(img_fp)
        print('\n')
        print_preds(pred)
        # assert cal_score(pred, expected) >= 0.8

        img = read_img(img_fp)
        pred = ocr.ocr(img)
        print_preds(pred)
        # assert cal_score(pred, expected) >= 0.8

        img = read_img(img_fp, gray=False)
        pred = ocr.ocr(img)
        print_preds(pred)
        # assert cal_score(pred, expected) >= 0.8


@pytest.mark.parametrize('img_fp, expected', SINGLE_LINE_CASES)
def test_ocr_for_single_line(img_fp, expected):
    ocr = CNOCR
    img_fp = os.path.join(example_dir, img_fp)
    pred = ocr.ocr_for_single_line(img_fp)
    print('\n')
    print_preds([pred])
    assert cal_score([pred], expected) >= 0.8

    img = read_img(img_fp)
    pred = ocr.ocr_for_single_line(img)
    print_preds([pred])
    assert cal_score([pred], expected) >= 0.8

    img = read_img(img_fp, gray=False)
    pred = ocr.ocr_for_single_line(img)
    print_preds([pred])
    assert cal_score([pred], expected) >= 0.8

    img = np.array(Image.fromarray(img).convert('L'))
    assert len(img.shape) == 2
    pred = ocr.ocr_for_single_line(img)
    print_preds([pred])
    assert cal_score([pred], expected) >= 0.8

    img = np.expand_dims(img, axis=2)
    assert len(img.shape) == 3 and img.shape[2] == 1
    pred = ocr.ocr_for_single_line(img)
    print_preds([pred])
    assert cal_score([pred], expected) >= 0.8


@pytest.mark.parametrize('img_fp, expected', MULTIPLE_LINE_CASES)
def test_ocr_for_single_lines(img_fp, expected):
    ocr = CNOCR
    img_fp = os.path.join(example_dir, img_fp)
    img = read_img(img_fp)
    if img.mean() < 145:  # 把黑底白字的图片对调为白底黑字
        img = 255 - img
    line_imgs = line_split(np.squeeze(img, -1), blank=True)
    line_img_list = [line_img for line_img, _ in line_imgs]
    pred = ocr.ocr_for_single_lines(line_img_list)
    print('\n')
    print_preds(pred)
    assert cal_score(pred, expected) >= 0.8

    line_img_list = [np.array(line_img) for line_img in line_img_list]
    pred = ocr.ocr_for_single_lines(line_img_list)
    print_preds(pred)
    assert cal_score(pred, expected) >= 0.8


def test_cand_alphabet():
    img_fp = os.path.join(example_dir, 'hybrid.png')

    ocr = CnOcr(
        'densenet_lite_136-fc', det_model_name='naive_det', cand_alphabet=NUMBERS
    )
    pt_pred = ocr.ocr(img_fp)
    pred = [line_p['text'] for line_p in pt_pred]
    print("Predicted Chars:", pred)
    assert len(pred) == 1 and pred[0] == '012345678'

    ocr = CnOcr(
        'densenet_lite_136-fc',
        rec_model_backend='onnx',
        det_model_name='naive_det',
        cand_alphabet=NUMBERS,
    )
    onnx_pred = ocr.ocr(img_fp)
    pred = [line_p['text'] for line_p in onnx_pred]
    print("Predicted Chars:", pred)
    assert len(pred) == 1 and pred[0] == '012345678'

    assert pt_pred[0]['text'] == onnx_pred[0]['text']
    assert abs(pt_pred[0]['score'] - onnx_pred[0]['score']) < 1e-5


@pytest.mark.parametrize('img_fp, expected', SINGLE_LINE_CASES)
def test_onnx(img_fp, expected):
    img_fp = os.path.join(example_dir, img_fp)

    pt_ocr = CnOcr(
        'densenet_lite_136-fc', rec_model_backend='pytorch', det_model_name='naive_det'
    )
    start_time = time.time()
    pt_preds = pt_ocr.ocr_for_single_line(img_fp)
    end_time = time.time()
    print(f'\npytorch time cost {end_time - start_time}', pt_preds)

    onnx_ocr = CnOcr(
        'densenet_lite_136-fc', rec_model_backend='onnx', det_model_name='naive_det'
    )
    start_time = time.time()
    onnx_preds = onnx_ocr.ocr_for_single_line(img_fp)
    end_time = time.time()
    print(f'onnx time cost {end_time - start_time}', onnx_preds, '\n\n')

    assert pt_preds['text'] == onnx_preds['text']
    assert abs(pt_preds['score'] - onnx_preds['score']) < 1e-5


@pytest.mark.parametrize('img_fp, expected', MULTIPLE_LINE_CASES)
def test_det_rec(img_fp, expected):
    ocr = CnOcr()
    img_fp = os.path.join(example_dir, img_fp)

    pred = ocr.ocr(img_fp)
    print('\n')
    print_preds(pred)
    # assert cal_score(pred, expected) >= 0.8

    img = read_img(img_fp)
    pred = ocr.ocr(img)
    print_preds(pred)
    # assert cal_score(pred, expected) >= 0.8

    img = read_img(img_fp, gray=False)
    pred = ocr.ocr(img)
    print_preds(pred)
    # assert cal_score(pred, expected) >= 0.8
