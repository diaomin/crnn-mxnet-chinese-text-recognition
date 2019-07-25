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
""" An example of predicting CAPTCHA image data with a LSTM network pre-trained with a CTC loss"""

from __future__ import print_function

import mxnet as mx
import numpy as np
# from PIL import Image
import argparse

from cnocr.fit.ctc_metrics import CtcMetrics
from cnocr.hyperparams.cn_hyperparams import CnHyperparams as Hyperparams
from cnocr.hyperparams.hyperparams2 import Hyperparams as Hyperparams2
from cnocr.fit.lstm import init_states
from cnocr.data_utils.data_iter import SimpleBatch
from cnocr.symbols.crnn import crnn_lstm


def read_captcha_img(path, hp):
    """ Reads image specified by path into numpy.ndarray"""
    import cv2
    tgt_h, tgt_w = hp.img_height, hp.img_width
    img = cv2.imread(path, 0)
    # import pdb; pdb.set_trace()
    # img = img.astype(np.float32) / 255.0
    img = cv2.resize(img, (tgt_w, tgt_h)).astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)  # res: [1, height, width]
    return img


def read_ocr_img(path, hp):
    # img = Image.open(path).resize((hp.img_width, hp.img_height), Image.BILINEAR)
    # img = img.convert('L')
    # img = np.expand_dims(np.array(img), 0)
    # return img
    img = mx.image.imread(path, 0)
    scale = hp.img_height / img.shape[0]
    new_width = int(scale * img.shape[1])
    hp._seq_length = new_width // 8
    img = mx.image.imresize(img, new_width, hp.img_height).asnumpy()
    img = np.squeeze(img, axis=2)
    # import pdb; pdb.set_trace()
    return np.expand_dims(img, 0)

    # img2 = mx.image.imread(path)
    # img2 = mx.image.imresize(img2, hp.img_width, hp.img_height)
    # img2 = cv2.cvtColor(img2.asnumpy(), cv2.COLOR_RGB2GRAY)
    # img2 = np.expand_dims(np.array(img2), 0)
    # return img2


def lstm_init_states(batch_size, hp):
    """ Returns a tuple of names and zero arrays for LSTM init states"""
    init_shapes = init_states(batch_size=batch_size, num_lstm_layer=hp.num_lstm_layer, num_hidden=hp.num_hidden)
    init_names = [s[0] for s in init_shapes]
    init_arrays = [mx.nd.zeros(x[1]) for x in init_shapes]
    # init_names.append('seq_length')
    # init_arrays.append(hp.seq_length)
    return init_names, init_arrays


def load_module(prefix, epoch, data_names, data_shapes, network=None):
    """
    Loads the model from checkpoint specified by prefix and epoch, binds it
    to an executor, and sets its parameters and returns a mx.mod.Module
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    if network is not None:
        sym = network

    # We don't need CTC loss for prediction, just a simple softmax will suffice.
    # We get the output of the layer just before the loss layer ('pred_fc') and add softmax on top
    pred_fc = sym.get_internals()['pred_fc_output']
    sym = mx.sym.softmax(data=pred_fc)

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=data_names, label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=False)
    return mod


def read_charset(charset_fp):
    alphabet = []
    # 第0个元素是预留id，在CTC中用来分割字符。它不对应有意义的字符
    with open(charset_fp) as fp:
        for line in fp:
            alphabet.append(line.rstrip('\n'))
    print('Alphabet size: %d' % len(alphabet))
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    inv_alph_dict[' '] = inv_alph_dict['<space>']  # 对应空格
    return alphabet, inv_alph_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="use which kind of dataset, captcha or cn_ocr",
                        choices=['captcha', 'cn_ocr'], type=str, default='captcha')
    parser.add_argument("--file", help="Path to the CAPTCHA image file")
    parser.add_argument("--prefix", help="Checkpoint prefix [Default 'ocr']", default='./models/model')
    parser.add_argument("--epoch", help="Checkpoint epoch [Default 100]", type=int, default=20)
    parser.add_argument('--charset_file', type=str, help='存储了每个字对应哪个id的关系.')
    args = parser.parse_args()
    if args.dataset == 'cn_ocr':
        hp = Hyperparams()
        img = read_ocr_img(args.file, hp)
    else:
        hp = Hyperparams2()
        img = read_captcha_img(args.file, hp)

    # init_state_names, init_state_arrays = lstm_init_states(batch_size=1, hp=hp)
    # import pdb; pdb.set_trace()

    sample = SimpleBatch(data_names=['data'], data=[mx.nd.array([img])])

    network = crnn_lstm(hp)
    mod = load_module(args.prefix, args.epoch, sample.data_names, sample.provide_data, network=network)

    mod.forward(sample)
    prob = mod.get_outputs()[0].asnumpy()

    prediction, start_end_idx = CtcMetrics.ctc_label(np.argmax(prob, axis=-1).tolist())

    if args.charset_file:
        alphabet, _ = read_charset(args.charset_file)
        res = [alphabet[p] for p in prediction]
        print("Predicted Chars:", res)
    else:
        # Predictions are 1 to 10 for digits 0 to 9 respectively (prediction 0 means no-digit)
        prediction = [p - 1 for p in prediction]
        print("Digits:", prediction)
    return


if __name__ == '__main__':
    main()
