# coding: utf-8
import os
import sys
import logging
from copy import deepcopy
import pytest
import mxnet as mx
from mxnet import nd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr.consts import EMB_MODEL_TYPES, SEQ_MODEL_TYPES
from cnocr.utils import set_logger
from cnocr.hyperparams.cn_hyperparams import CnHyperparams
from cnocr.symbols.densenet import _make_dense_layer, DenseNet, cal_num_params
from cnocr.symbols.crnn import (
    CRnn,
    pipline,
    gen_network,
    get_infer_shape,
    crnn_lstm,
    crnn_lstm_lite,
)

logger = set_logger('info')

HP = CnHyperparams()


def test_dense_layer():
    x = nd.random.randn(128, 64, 32, 280)
    net = _make_dense_layer(64, 2, 0.1)
    net.initialize()
    y = net(x)
    logger.info(net)
    logger.info(y.shape)


def test_densenet():
    x = nd.random.randn(128, 64, 32, 280)
    layer_channels = (64, 128, 256, 512)
    net = DenseNet(layer_channels)
    net.initialize()
    y = net(x)
    logger.info(net)
    logger.info(y.shape)  # (128, 512, 1, 69)
    assert y.shape[2] == 1
    logger.info('number of parameters: %d', cal_num_params(net))  # 1748224


def test_crnn():
    _hp = deepcopy(HP)
    _hp.set_seq_length(_hp.img_width // 4)
    x = nd.random.randn(128, 64, 32, 280)
    layer_channels_list = [(64, 128, 256, 512), (32, 64, 128, 256)]
    for layer_channels in layer_channels_list:
        densenet = DenseNet(layer_channels)
        crnn = CRnn(_hp, densenet)
        crnn.initialize()
        y = crnn(x)
        logger.info(
            'output shape: %s', y.shape
        )  # res: `(sequence_length, batch_size, 2*num_hidden)`
        assert y.shape == (_hp.seq_length, _hp.batch_size, 2 * _hp.num_hidden)
        logger.info('number of parameters: %d', cal_num_params(crnn))


def test_crnn_lstm():
    hp = deepcopy(HP)
    hp.set_seq_length(hp.img_width // 8)
    data = mx.sym.Variable('data', shape=(128, 1, 32, 280))
    pred = crnn_lstm(HP, data)
    pred_shape = pred.infer_shape()[1][0]
    logger.info('shape of pred: %s', pred_shape)
    assert pred_shape == (hp.seq_length, hp.batch_size, 2 * hp.num_hidden)


def test_crnn_lstm_lite():
    hp = deepcopy(HP)
    hp.set_seq_length(hp.img_width // 4 - 1)
    data = mx.sym.Variable('data', shape=(128, 1, 32, 280))
    pred = crnn_lstm_lite(HP, data)
    pred_shape = pred.infer_shape()[1][0]
    logger.info('shape of pred: %s', pred_shape)
    assert pred_shape == (hp.seq_length, hp.batch_size, 2 * hp.num_hidden)


def test_pipline():
    hp = deepcopy(HP)
    hp.set_seq_length(hp.img_width // 4 - 1)
    hp._loss_type = None  # infer mode
    layer_channels_list = [(64, 128, 256, 512), (32, 64, 128, 256)]
    for layer_channels in layer_channels_list:
        densenet = DenseNet(layer_channels)
        crnn = CRnn(hp, densenet)
        data = mx.sym.Variable('data', shape=(128, 1, 32, 280))
        pred = pipline(crnn, hp, data)
        pred_shape = pred.infer_shape()[1][0]
        logger.info('shape of pred: %s', pred_shape)
        assert pred_shape == (hp.batch_size * hp.seq_length, hp.num_classes)


MODEL_NAMES = []
for emb_model in EMB_MODEL_TYPES:
    for seq_model in SEQ_MODEL_TYPES:
        MODEL_NAMES.append('%s-%s' % (emb_model, seq_model))


@pytest.mark.parametrize(
    'model_name', MODEL_NAMES
)
def test_gen_networks(model_name):
    logger.info('model_name: %s', model_name)
    network, hp = gen_network(model_name, HP)
    shape_dict = get_infer_shape(network, HP)
    logger.info('shape_dict: %s', shape_dict)
    assert shape_dict['pred_fc_output'] == (
        hp.batch_size * hp.seq_length,
        hp.num_classes,
    )
