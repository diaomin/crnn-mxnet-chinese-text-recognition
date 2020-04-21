# coding: utf-8
import os
import sys
import pytest
import numpy as np
import mxnet as mx
from mxnet import nd
from PIL import Image
import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr import CnOcr
from cnocr.line_split import line_split
from cnocr.data_utils.aug import GrayAug

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
example_dir = os.path.join(root_dir, 'examples')
CNOCR = CnOcr(model_name='conv-lite-fc', model_epoch=None)

SINGLE_LINE_CASES = [
    ('20457890_2399557098.jpg', ['就会哈哈大笑。3.0']),
    ('rand_cn1.png', ['笠淡嘿骅谧鼎皋姚歼蠢驼耳胬挝涯狗蒽孓犷']),
    ('rand_cn2.png', ['凉芦']),
    ('helloworld.jpg', ['Hello World!你好世界']),
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
    pred = [''.join(line_p) for line_p in pred]
    print("Predicted Chars:", pred)


def cal_score(preds, expected):
    if len(preds) != len(expected):
        return 0
    total_cnt = 0
    total_dist = 0
    for real, pred in zip(expected, preds):
        pred = ''.join(pred)
        distance = Levenshtein.distance(real, pred)
        total_dist += distance
        total_cnt += len(real)

    return 1.0 - float(total_dist) / total_cnt


@pytest.mark.parametrize('img_fp, expected', CASES)
def test_ocr(img_fp, expected):
    ocr = CNOCR
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_fp = os.path.join(root_dir, 'examples', img_fp)
    # img_fp = 'multi-line-game.jpeg'

    pred = ocr.ocr(img_fp)
    print('\n')
    print_preds(pred)
    assert cal_score(pred, expected) >= 0.9

    img = mx.image.imread(img_fp, 1)
    pred = ocr.ocr(img)
    print_preds(pred)
    assert cal_score(pred, expected) >= 0.9

    img = mx.image.imread(img_fp, 1).asnumpy()
    pred = ocr.ocr(img)
    print_preds(pred)
    assert cal_score(pred, expected) >= 0.9


@pytest.mark.parametrize('img_fp, expected', SINGLE_LINE_CASES)
def test_ocr_for_single_line(img_fp, expected):
    ocr = CNOCR
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_fp = os.path.join(root_dir, 'examples', img_fp)
    pred = ocr.ocr_for_single_line(img_fp)
    print('\n')
    print_preds(pred)
    assert cal_score([pred], expected) >= 0.9

    img = mx.image.imread(img_fp, 1)
    pred = ocr.ocr_for_single_line(img)
    print_preds(pred)
    assert cal_score([pred], expected) >= 0.9

    img = mx.image.imread(img_fp, 1).asnumpy()
    pred = ocr.ocr_for_single_line(img)
    print_preds(pred)
    assert cal_score([pred], expected) >= 0.9

    img = np.array(Image.fromarray(img).convert('L'))
    assert len(img.shape) == 2
    pred = ocr.ocr_for_single_line(img)
    print_preds(pred)
    assert cal_score([pred], expected) >= 0.9

    img = np.expand_dims(img, axis=2)
    assert len(img.shape) == 3 and img.shape[2] == 1
    pred = ocr.ocr_for_single_line(img)
    print_preds(pred)
    assert cal_score([pred], expected) >= 0.9


@pytest.mark.parametrize('img_fp, expected', MULTIPLE_LINE_CASES)
def test_ocr_for_single_lines(img_fp, expected):
    ocr = CNOCR
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_fp = os.path.join(root_dir, 'examples', img_fp)
    img = mx.image.imread(img_fp, 1).asnumpy()
    if img.mean() < 145:  # 把黑底白字的图片对调为白底黑字
        img = 255 - img
    line_imgs = line_split(img, blank=True)
    line_img_list = [line_img for line_img, _ in line_imgs]
    pred = ocr.ocr_for_single_lines(line_img_list)
    print('\n')
    print_preds(pred)
    assert cal_score(pred, expected) >= 0.9

    line_img_list = [nd.array(line_img) for line_img in line_img_list]
    pred = ocr.ocr_for_single_lines(line_img_list)
    print_preds(pred)
    assert cal_score(pred, expected) >= 0.9


@pytest.mark.parametrize('img_fp, expected', SINGLE_LINE_CASES)
def test_gray_aug(img_fp, expected):
    img_fp = os.path.join(example_dir, img_fp)
    img = mx.image.imread(img_fp, 1)
    aug = GrayAug()
    res_img = aug(img)
    print(res_img.shape, res_img.dtype)


def test_cand_alphabet():
    from cnocr.consts import NUMBERS

    img_fp = os.path.join(example_dir, 'hybrid.png')

    ocr = CnOcr()
    pred = ocr.ocr(img_fp)
    pred = [''.join(line_p) for line_p in pred]
    print("Predicted Chars:", pred)
    assert len(pred) == 1 and pred[0] == 'o12345678'

    ocr = CnOcr(cand_alphabet=NUMBERS)
    pred = ocr.ocr(img_fp)
    pred = [''.join(line_p) for line_p in pred]
    print("Predicted Chars:", pred)
    assert len(pred) == 1 and pred[0] == '012345678'
