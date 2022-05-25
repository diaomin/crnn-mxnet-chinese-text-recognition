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
import sys
import logging
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr.utils import set_logger, read_img
from cnocr.consts import NUMBERS
from cnocr.ppocr import PPRecognizer

logger = set_logger(log_level=logging.INFO)

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
example_dir = os.path.join(root_dir, 'docs/examples')
CNOCR = PPRecognizer(model_name='ch_PP-OCRv3')

SINGLE_LINE_CASES = [
    ('20457890_2399557098.jpg', ['就会哈哈大笑。3.0']),
    # ('rand_cn1.png', ['笠淡嘿骅谧鼎皋姚歼蠢驼耳胬挝涯狗蒽孓犷']),
    # ('rand_cn2.png', ['凉芦']),
    ('helloworld.jpg', ['Hello world!你好世界']),
    ('hybrid.png', ['o12345678']),
]


def print_preds(pred):
    pred = pred[0]
    print("Predicted Chars:", pred)


def cal_score(preds, expected):
    import Levenshtein

    if len(preds) != len(expected):
        return 0
    total_cnt = 0
    total_dist = 0
    for real, (pred, _) in zip(expected, preds):
        pred = ''.join(pred)
        distance = Levenshtein.distance(real, pred)
        total_dist += distance
        total_cnt += len(real)

    return 1.0 - float(total_dist) / total_cnt


@pytest.mark.parametrize('img_fp, expected', SINGLE_LINE_CASES)
def test_ppocr(img_fp, expected):
    ocr = CNOCR
    img_fp = os.path.join(example_dir, img_fp)

    pred = ocr.recognize([img_fp])[0]
    print_preds(pred)
    assert cal_score([pred], expected) >= 0.8

    img = read_img(img_fp, gray=False)
    pred = ocr.recognize([img])[0]
    print_preds(pred)
    assert cal_score([pred], expected) >= 0.8

    img = read_img(img_fp, gray=True)
    pred = ocr.recognize([img])[0]
    print_preds(pred)
    assert cal_score([pred], expected) >= 0.8


def test_cand_alphabet():
    img_fp = os.path.join(example_dir, 'hybrid.png')

    ocr = PPRecognizer(model_name='en_number_mobile_v2.0', cand_alphabet=NUMBERS)
    # ocr = PPRecognizer(model_name='ch_PP-OCRv3', cand_alphabet=NUMBERS)
    img = read_img(img_fp, gray=False)
    pred = ocr.recognize([img])[0]
    print("Predicted Chars:", pred)
    assert pred[0] == '012345678'
