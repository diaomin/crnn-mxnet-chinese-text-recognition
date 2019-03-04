# coding:utf-8
from __future__ import print_function

import argparse
import logging
import os
import mxnet as mx
from data_utils.captcha_generator import MPDigitCaptcha

# from hyperparams.hyperparams import Hyperparams
from hyperparams.hyperparams2 import Hyperparams
from data_utils.data_iter import ImageIterLstm, OCRIter
from symbols.crnn import crnn_no_lstm, crnn_lstm
from fit.ctc_metrics import CtcMetrics
from fit.fit import fit

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", help="Path to image files", type=str,
                        default='/home/richard/data/Synthetic_Chinese_String_Dataset/images')
    parser.add_argument("--train_file", help="Path to train txt file", type=str,
                        default='/home/richard/data/Synthetic_Chinese_String_Dataset/train.txt')
    parser.add_argument("--test_file", help="Path to test txt file", type=str,
                        default='/home/richard/data/Synthetic_Chinese_String_Dataset/test.txt')
    parser.add_argument("--cpu",
                        help="Number of CPUs for training [Default 8]. Ignored if --gpu is specified.",
                        type=int, default=2)
    parser.add_argument("--gpu", help="Number of GPUs for training [Default 0]", type=int)
    parser.add_argument('--load_epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    parser.add_argument("--prefix", help="Checkpoint prefix [Default 'ocr']", default='./check_points/model')
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


def main():
    args = parse_args()
    hp = Hyperparams()

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
    # print(img.shape)
    # import numpy as np
    # import cv2
    # img = np.transpose(img, (1, 0))
    # cv2.imwrite('xxx.png', img * 255)
    # import pdb; pdb.set_trace()

    init_c = [('l%d_init_c' % l, (hp.batch_size, hp.num_hidden)) for l in range(hp.num_lstm_layer * 2)]
    init_h = [('l%d_init_h' % l, (hp.batch_size, hp.num_hidden)) for l in range(hp.num_lstm_layer * 2)]
    init_states = init_c + init_h
    data_names = ['data'] + [x[0] for x in init_states]

    # data_train = ImageIterLstm(
    #     args.data_root, args.train_file, hp.batch_size, (hp.img_width, hp.img_height), hp.num_label, init_states, name="train")
    # data_val = ImageIterLstm(
    #     args.data_root, args.test_file,  hp.batch_size, (hp.img_width, hp.img_height), hp.num_label, init_states, name="val")
    data_train = OCRIter(
        hp.train_epoch_size // hp.batch_size, hp.batch_size, init_states, captcha=mp_captcha, name='train')
    data_val = OCRIter(
        hp.eval_epoch_size // hp.batch_size, hp.batch_size, init_states, captcha=mp_captcha, name='val')
    data_train.__iter__()

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    metrics = CtcMetrics(hp.seq_length)

    fit(network=network, data_train=data_train, data_val=data_val, metrics=metrics, args=args, hp=hp, data_names=data_names)

    mp_captcha.reset()


if __name__ == '__main__':
    main()
