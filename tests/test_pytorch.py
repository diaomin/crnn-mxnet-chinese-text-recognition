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

import torch

from cnocr.consts import IMG_STANDARD_HEIGHT, ENG_LETTERS, VOCAB_FP
from cnocr.utils import read_charset, pad_img_seq, load_model_params, read_img, rescale_img, normalize_img_array
from cnocr.models.densenet import DenseNet
from cnocr.models.ocr_model import OcrModel


def test_crnn():
    width = 280
    img = torch.rand(4, 1, IMG_STANDARD_HEIGHT, width)
    net = DenseNet(32, [2, 2, 2, 2], 64)
    net.eval()
    res = net(img)
    print(res.shape)

    crnn = OcrModel(net, vocab=ENG_LETTERS, lstm_features=512, rnn_units=128)
    res2 = crnn(img)
    print(res2)


def test_crnn_for_variable_length():
    vocab, letter2id = read_charset(VOCAB_FP)
    net = DenseNet(32, [2, 2, 2, 2], 64)
    crnn = OcrModel(net, vocab=vocab, lstm_features=512, rnn_units=128)
    crnn.eval()
    model_fp = VOCAB_FP.parent / 'models/last.ckpt'
    if model_fp.exists():
        print(f'load model params from {model_fp}')
        load_model_params(crnn, model_fp)
    width = 280
    img1 = torch.rand(1, IMG_STANDARD_HEIGHT, width)
    img2 = torch.rand(1, IMG_STANDARD_HEIGHT, width // 2)
    img3 = torch.rand(1, IMG_STANDARD_HEIGHT, width * 2)
    imgs = pad_img_seq([img1, img2, img3])
    input_lengths = torch.Tensor([width, width // 2, width * 2])
    out = crnn(
        imgs, input_lengths=input_lengths, return_model_output=True, return_preds=True,
    )
    print(out['preds'])

    padded = torch.zeros((3, 1, IMG_STANDARD_HEIGHT, 50))
    imgs2 = torch.cat((imgs, padded), dim=-1)
    out2 = crnn(
        imgs2, input_lengths=input_lengths, return_model_output=True, return_preds=True,
    )
    print(out2['preds'])
    # breakpoint()


def test_crnn_for_variable_length2():
    vocab, letter2id = read_charset(VOCAB_FP)
    net = DenseNet(32, [2, 2, 2, 2], 64)
    crnn = OcrModel(net, vocab=vocab, lstm_features=512, rnn_units=128)
    crnn.eval()
    model_fp = VOCAB_FP.parent / 'models/last.ckpt'
    if model_fp.exists():
        print(f'load model params from {model_fp}')
        load_model_params(crnn, model_fp)
    img_fps = ('helloworld.jpg', 'helloworld-ch.jpg')
    imgs = []
    input_lengths = []
    for fp in img_fps:
        img = read_img(VOCAB_FP.parent / 'examples' / fp)
        img = rescale_img(img)
        input_lengths.append(img.shape[2])
        imgs.append(normalize_img_array(img))
    imgs = pad_img_seq(imgs)
    input_lengths = torch.Tensor(input_lengths)
    out = crnn(
        imgs, input_lengths=input_lengths, return_model_output=True, return_preds=True,
    )
    print(out['preds'])

    padded = torch.zeros((2, 1, IMG_STANDARD_HEIGHT, 80))
    imgs2 = torch.cat((imgs, padded), dim=-1)
    out2 = crnn(
        imgs2, input_lengths=input_lengths, return_model_output=True, return_preds=True,
    )
    print(out2['preds'])
    # breakpoint()
