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

from cnocr.consts import IMG_STANDARD_HEIGHT, ENG_LETTERS
from cnocr.models.ocr_model import OcrModel


def test_crnn():
    width = 280
    img = torch.rand(4, 1, IMG_STANDARD_HEIGHT, width)

    crnn = OcrModel.from_name('densenet_lite_136-fc', ENG_LETTERS)
    res2 = crnn(img, input_lengths=torch.tensor([width]))

    net = crnn.encoder
    net.eval()
    res = net(img)
    print(res.shape)

    print(res2)
