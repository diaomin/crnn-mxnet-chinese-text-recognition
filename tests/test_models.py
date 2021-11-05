# coding: utf-8
import os
import sys
from copy import deepcopy
import pytest
import torch
from torch import nn
from torchvision.models import (
    resnet50,
    resnet34,
    resnet18,
    mobilenet_v3_large,
    mobilenet_v3_small,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr.utils import set_logger, get_model_size
from cnocr.consts import IMG_STANDARD_HEIGHT, ENCODER_CONFIGS, DECODER_CONFIGS
from cnocr.models.densenet import DenseNet, DenseNetLite
from cnocr.models.mobilenet import gen_mobilenet_v3


logger = set_logger('info')


def test_conv():
    conv = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False)
    input = torch.rand(1, 32, 10, 4)
    res = conv(input)
    logger.info(res.shape)


def test_densenet():
    width = 280
    img = torch.rand(4, 1, IMG_STANDARD_HEIGHT, width)
    net = DenseNet(32, [2, 2, 2, 2], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  # 406464
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 128, 4, 35)

    net = DenseNet(32, [1, 1, 1, 4], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  # 301440
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    # assert tuple(res.shape) == (4, 100, 4, 35)
    #
    # net = DenseNet(32, [1, 1, 2, 2], 64)
    # net.eval()
    # logger.info(net)
    # logger.info(f'model size: {get_model_size(net)}')  # 243616
    # logger.info(img.shape)
    # res = net(img)
    # logger.info(res.shape)
    # assert tuple(res.shape) == (4, 116, 4, 35)
    #
    # net = DenseNet(32, [1, 2, 2, 2], 64)
    # net.eval()
    # logger.info(net)
    # logger.info(f'model size: {get_model_size(net)}')  # 230680
    # logger.info(img.shape)
    # res = net(img)
    # logger.info(res.shape)
    # assert tuple(res.shape) == (4, 124, 4, 35)
    #
    # net = DenseNet(32, [1, 1, 2, 4], 64)
    # net.eval()
    # logger.info(net)
    # logger.info(f'model size: {get_model_size(net)}')  # 230680
    # logger.info(img.shape)
    # res = net(img)
    # logger.info(res.shape)
    # assert tuple(res.shape) == (4, 180, 4, 35)


def test_densenet_lite():
    width = 280
    img = torch.rand(4, 1, IMG_STANDARD_HEIGHT, width)
    # net = DenseNetLite(32, [2, 2, 2], 64)
    # net.eval()
    # logger.info(net)
    # logger.info(f'model size: {get_model_size(net)}')  # 302976
    # logger.info(img.shape)
    # res = net(img)
    # logger.info(res.shape)
    # assert tuple(res.shape) == (4, 128, 2, 35)

    # net = DenseNetLite(32, [2, 1, 1], 64)
    # net.eval()
    # logger.info(net)
    # logger.info(f'model size: {get_model_size(net)}')  # 197952
    # logger.info(img.shape)
    # res = net(img)
    # logger.info(res.shape)
    # assert tuple(res.shape) == (4, 80, 2, 35)

    net = DenseNetLite(32, [1, 3, 4], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  # 186672
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 200, 2, 35)

    net = DenseNetLite(32, [1, 3, 6], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  # 186672
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 264, 2, 35)

    # net = DenseNetLite(32, [1, 2, 2], 64)
    # net.eval()
    # logger.info(net)
    # logger.info(f'model size: {get_model_size(net)}')  #
    # logger.info(img.shape)
    # res = net(img)
    # logger.info(res.shape)
    # assert tuple(res.shape) == (4, 120, 2, 35)


def test_mobilenet():
    width = 280
    img = torch.rand(4, 1, IMG_STANDARD_HEIGHT, width)
    net = gen_mobilenet_v3('tiny')
    net.eval()
    logger.info(net)
    res = net(img)
    logger.info(f'model size: {get_model_size(net)}')  # 186672
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 192, 2, 35)

    net = gen_mobilenet_v3('small')
    net.eval()
    logger.info(net)
    res = net(img)
    logger.info(f'model size: {get_model_size(net)}')  # 186672
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 192, 2, 35)

MODEL_NAMES = []
for emb_model in ENCODER_CONFIGS:
    for seq_model in DECODER_CONFIGS:
        MODEL_NAMES.append('%s-%s' % (emb_model, seq_model))


# @pytest.mark.parametrize(
#     'model_name', MODEL_NAMES
# )
# def test_gen_networks(model_name):
#     logger.info('model_name: %s', model_name)
#     network, hp = gen_network(model_name, HP)
#     shape_dict = get_infer_shape(network, HP)
#     logger.info('shape_dict: %s', shape_dict)
#     assert shape_dict['pred_fc_output'] == (
#         hp.batch_size * hp.seq_length,
#         hp.num_classes,
#     )
