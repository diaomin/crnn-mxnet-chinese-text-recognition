from __future__ import print_function

import argparse
import logging
import os
import mxnet as mx

from hyperparams.hyperparams import Hyperparams
from data_utils.data_iter import ImageIter,ImageIterLstm
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
                        type=int, default=4)
    parser.add_argument("--gpu", help="Number of GPUs for training [Default 0]", type=int, default=1)
    parser.add_argument('--load_epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    parser.add_argument("--prefix", help="Checkpoint prefix [Default 'ocr']", default='./check_points/model')
    return parser.parse_args()

def main():
    args = parse_args()
    hp = Hyperparams()

    init_c = [('l%d_init_c' % l, (hp.batch_size, hp.num_hidden)) for l in range(hp.num_lstm_layer * 2)]
    init_h = [('l%d_init_h' % l, (hp.batch_size, hp.num_hidden)) for l in range(hp.num_lstm_layer * 2)]
    init_states = init_c + init_h
    data_names = ['data'] + [x[0] for x in init_states]

    data_train = ImageIterLstm(
        args.data_root, args.train_file, hp.batch_size, (hp.img_width, hp.img_height), hp.num_label, init_states, name="train")
    data_val = ImageIterLstm(
        args.data_root, args.test_file,  hp.batch_size, (hp.img_width, hp.img_height), hp.num_label, init_states, name="val")

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    network = crnn_lstm(hp)

    metrics = CtcMetrics(hp.seq_length)

    fit(network=network, data_train=data_train, data_val=data_val, metrics=metrics, args=args, hp=hp, data_names=data_names)


if __name__ == '__main__':
    main()
