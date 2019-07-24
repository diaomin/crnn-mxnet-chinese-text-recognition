# coding: utf-8
import os
import sys
import pytest
import numpy as np
import mxnet as mx
from mxnet import nd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr import CnOcr
from cnocr.line_split import line_split

CNOCR = CnOcr()

SINGLE_LINE_CASES = [
    ('20457890_2399557098.jpg', [['就', '会', '哈', '哈', '大', '笑', '。', '3', '.', '0']]),
    ('rand_cn1.png', [['笠', '淡', '嘿', '骅', '谧', '鼎', '臭', '姚', '歼', '蠢', '驼', '耳', '裔', '挝', '涯', '狗', '蒽', '子', '犷']])
]
MULTIPLE_LINE_CASES = [
    ('multi-line_cn1.png', [['网', '络', '支', '付', '并', '无', '本', '质', '的', '区', '别', '，', '因', '为'],
                            ['每', '一', '个', '手', '机', '号', '码', '和', '邮', '件', '地', '址', '背', '后'],
                            ['都', '会', '对', '应', '着', '一', '个', '账', '户', '一', '―', '这', '个', '账'],
                            ['户', '可', '以', '是', '信', '用', '卡', '账', '户', '、', '借', '记', '卡', '账'],
                            ['户', '，', '也', '包', '括', '邮', '局', '汇', '款', '、', '手', '机', '代'],
                            ['收', '、', '电', '话', '代', '收', '、', '预', '付', '费', '卡', '和', '点', '卡'],
                            ['等', '多', '种', '形', '式', '。']]),
    ('multi-line_cn2.png', [['。', '当', '然', '，', '在', '媒', '介', '越', '来', '越', '多', '的', '情', '形', '下', '，'],
                            ['意', '味', '着', '传', '播', '方', '式', '的', '变', '化', '。', '过', '去', '主', '流'],
                            ['的', '是', '大', '众', '传', '播', '，', '现', '在', '互', '动', '性', '和', '定', '制'],
                            ['性', '带', '来', '了', '新', '的', '挑', '战', '—', '—', '如', '何', '让', '品', '牌'],
                            ['与', '消', '费', '者', '更', '加', '互', '动', '。']]),
]
CASES = SINGLE_LINE_CASES + MULTIPLE_LINE_CASES


@pytest.mark.parametrize('img_fp, expected', CASES)
def test_ocr(img_fp, expected):
    ocr = CNOCR
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_fp = os.path.join(root_dir, 'examples', img_fp)
    pred = ocr.ocr(img_fp)
    print('\n')
    print("Predicted Chars:", pred)
    assert expected == pred
    img = mx.image.imread(img_fp, 1)
    pred = ocr.ocr(img)
    print("Predicted Chars:", pred)
    assert expected == pred
    img = mx.image.imread(img_fp, 1).asnumpy()
    pred = ocr.ocr(img)
    print("Predicted Chars:", pred)
    assert expected == pred


@pytest.mark.parametrize('img_fp, expected', SINGLE_LINE_CASES)
def test_ocr_for_single_line(img_fp, expected):
    ocr = CNOCR
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_fp = os.path.join(root_dir, 'examples', img_fp)
    pred = ocr.ocr_for_single_line(img_fp)
    print('\n')
    print("Predicted Chars:", pred)
    assert expected[0] == pred
    img = mx.image.imread(img_fp, 1)
    pred = ocr.ocr_for_single_line(img)
    print("Predicted Chars:", pred)
    assert expected[0]== pred
    img = mx.image.imread(img_fp, 1).asnumpy()
    pred = ocr.ocr_for_single_line(img)
    print("Predicted Chars:", pred)
    assert expected[0] == pred
    img = np.array(Image.fromarray(img).convert('L'))
    assert len(img.shape) == 2
    pred = ocr.ocr_for_single_line(img)
    print("Predicted Chars:", pred)
    assert expected[0] == pred
    img = np.expand_dims(img, axis=2)
    assert len(img.shape) == 3 and img.shape[2] == 1
    pred = ocr.ocr_for_single_line(img)
    print("Predicted Chars:", pred)
    assert expected[0] == pred


@pytest.mark.parametrize('img_fp, expected', MULTIPLE_LINE_CASES)
def test_ocr_for_single_lines(img_fp, expected):
    ocr = CNOCR
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_fp = os.path.join(root_dir, 'examples', img_fp)
    img = mx.image.imread(img_fp, 1).asnumpy()
    line_imgs = line_split(img, blank=True)
    line_img_list = [line_img for line_img, _ in line_imgs]
    pred = ocr.ocr_for_single_lines(line_img_list)
    print('\n')
    print("Predicted Chars:", pred)
    assert expected == pred
    line_img_list = [nd.array(line_img) for line_img in line_img_list]
    pred = ocr.ocr_for_single_lines(line_img_list)
    print("Predicted Chars:", pred)
    assert expected == pred
