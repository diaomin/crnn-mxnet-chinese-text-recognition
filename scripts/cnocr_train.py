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
from __future__ import print_function

import argparse
import logging
import os
import sys
import mxnet as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnocr.__version__ import __version__
from cnocr.utils import data_dir
from cnocr.hyperparams.cn_hyperparams import CnHyperparams as Hyperparams
from cnocr.hyperparams.hyperparams2 import Hyperparams as Hyperparams2
from cnocr.data_utils.data_iter import ImageIterLstm, MPOcrImages, OCRIter, GrayImageIter
from cnocr.data_utils.aug import FgBgFlipAug
from cnocr.symbols.crnn import crnn_no_lstm, crnn_lstm
from cnocr.fit.ctc_metrics import CtcMetrics
from cnocr.fit.fit import fit


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    default_model_prefix = os.path.join(data_dir(), 'models', 'model-v{}'.format(__version__))

    parser.add_argument("--dataset",
                        help="use which kind of dataset, captcha or cn_ocr",
                        choices=['captcha', 'cn_ocr'],
                        type=str, default='captcha')
    parser.add_argument("--data_root", help="Path to image files", type=str,
                        default='/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/wechat_simulator')
    parser.add_argument("--train_file", help="Path to train txt file", type=str,
                        default='/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/wechat_simulator/train.txt')
    parser.add_argument("--test_file", help="Path to test txt file", type=str,
                        default='/Users/king/Documents/WhatIHaveDone/Test/text_renderer/output/wechat_simulator/test.txt')
    parser.add_argument("--cpu",
                        help="Number of CPUs for training [Default 8]. Ignored if --gpu is specified.",
                        type=int, default=2)
    parser.add_argument("--gpu", help="Number of GPUs for training [Default 0]", type=int)
    parser.add_argument('--load_epoch', type=int,
                        help='load the model on an epoch using the model-load-prefix [Default: no trained model will be loaded]')
    parser.add_argument("--prefix", help="Checkpoint prefix [Default '{}']".format(default_model_prefix),
                        default=default_model_prefix)
    parser.add_argument("--loss", help="'ctc' or 'warpctc' loss [Default 'ctc']", default='ctc')
    parser.add_argument("--num_proc", help="Number CAPTCHA generating processes [Default 4]", type=int, default=4)
    parser.add_argument("--font_path", help="Path to ttf font file or directory containing ttf files")
    return parser.parse_args()


def get_fonts(path):
    fonts = list()
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith('.ttf') or filename.endswith('.ttc'):
                fonts.append(os.path.join(path, filename))
    else:
        fonts.append(path)
    return fonts


def run_captcha(args):
    from cnocr.data_utils.captcha_generator import MPDigitCaptcha

    hp = Hyperparams2()

    network = crnn_lstm(hp)
    # arg_shape, out_shape, aux_shape = network.infer_shape(data=(128, 1, 32, 100), label=(128, 10),
    #                                                       l0_init_h=(128, 100), l1_init_h=(128, 100), l2_init_h=(128, 100), l3_init_h=(128, 100))
    # print(dict(zip(network.list_arguments(), arg_shape)))
    # import pdb; pdb.set_trace()

    # Start a multiprocessor captcha image generator
    mp_captcha = MPDigitCaptcha(
        font_paths=get_fonts(args.font_path), h=hp.img_width, w=hp.img_height,
        num_digit_min=3, num_digit_max=4, num_processes=args.num_proc, max_queue_size=hp.batch_size * 2)
    mp_captcha.start()
    # img, num = mp_captcha.get()
    # print(img.shape, num)
    # import numpy as np
    # import cv2
    # img = np.transpose(img, (1, 0))
    # cv2.imwrite('captcha1.png', img * 255)
    # import sys
    # sys.exit(0)
    # import pdb; pdb.set_trace()

    # init_c = [('l%d_init_c' % l, (hp.batch_size, hp.num_hidden)) for l in range(hp.num_lstm_layer * 2)]
    # init_h = [('l%d_init_h' % l, (hp.batch_size, hp.num_hidden)) for l in range(hp.num_lstm_layer * 2)]
    # init_states = init_c + init_h
    # data_names = ['data'] + [x[0] for x in init_states]
    data_names = ['data']

    data_train = OCRIter(
        hp.train_epoch_size // hp.batch_size, hp.batch_size, captcha=mp_captcha, num_label=hp.num_label,
        name='train')
    data_val = OCRIter(
        hp.eval_epoch_size // hp.batch_size, hp.batch_size, captcha=mp_captcha, num_label=hp.num_label,
        name='val')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    metrics = CtcMetrics(hp.seq_length)

    fit(network=network, data_train=data_train, data_val=data_val, metrics=metrics, args=args, hp=hp, data_names=data_names)

    mp_captcha.reset()


def run_cn_ocr(args):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    hp = Hyperparams()

    network = crnn_lstm(hp)
    metrics = CtcMetrics(hp.seq_length)

    # mp_data_train = MPOcrImages(args.data_root, args.train_file, (hp.img_width, hp.img_height), hp.num_label,
    #                             num_processes=args.num_proc, max_queue_size=hp.batch_size * 100)
    # mp_data_test = MPOcrImages(args.data_root, args.test_file, (hp.img_width, hp.img_height), hp.num_label,
    #                            num_processes=max(args.num_proc // 2, 1), max_queue_size=hp.batch_size * 10)
    # mp_data_train.start()
    # mp_data_test.start()

    # data_train = OCRIter(
    #     hp.train_epoch_size // hp.batch_size, hp.batch_size, captcha=mp_data_train, num_label=hp.num_label,
    #     name='train')
    # data_val = OCRIter(
    #     hp.eval_epoch_size // hp.batch_size, hp.batch_size, captcha=mp_data_test, num_label=hp.num_label,
    #     name='val')
    data_train, data_val = _gen_iters(hp, args.train_file, args.test_file)
    data_names = ['data']
    fit(network=network, data_train=data_train, data_val=data_val, metrics=metrics, args=args, hp=hp, data_names=data_names)

    # mp_data_train.reset()
    # mp_data_test.reset()


def _gen_iters(hp, train_fp_prefix, val_fp_prefix):
    height, width = hp.img_height, hp.img_width
    augs = mx.image.CreateAugmenter(
        data_shape=(3, height, width),
        resize=0,
        rand_crop=False,
        rand_resize=False,
        rand_mirror=False,
        mean=None,
        std=None,
        brightness=0.001,
        contrast=0.001,
        saturation=0.001,
        hue=0.05,
        pca_noise=0.1,
        inter_method=2,
    )
    augs.append(FgBgFlipAug(p=0.2))
    train_iter = GrayImageIter(
        batch_size=hp.batch_size,
        data_shape=(3, height, width),
        label_width=hp.num_label,
        dtype='int32',
        shuffle=True,
        path_imgrec=str(train_fp_prefix) + ".rec",
        path_imgidx=str(train_fp_prefix) + ".idx",
        aug_list=augs,
    )

    val_iter = GrayImageIter(
        batch_size=hp.batch_size,
        data_shape=(3, height, width),
        label_width=hp.num_label,
        dtype='int32',
        path_imgrec=str(val_fp_prefix) + ".rec",
        path_imgidx=str(val_fp_prefix) + ".idx",
    )

    return train_iter, val_iter


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == 'captcha':
        run_captcha(args)
    else:
        run_cn_ocr(args)
