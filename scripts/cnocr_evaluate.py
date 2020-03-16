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

import sys
import os
import argparse
import mxnet as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnocr import CnOcr
from cnocr.consts import MODEL_NAMES


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="model name",
        choices=MODEL_NAMES,
        type=str,
        default='conv-rnn',
    )
    parser.add_argument(
        "-i",
        "--input-fp",
        default='test.txt',
        help="the file path with image names and labels",
    )
    parser.add_argument(
        "--image-prefix-dir", default='.', help="图片所在文件夹，相对于索引文件中记录的图片位置"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help="whether to print details to screen",
    )

    parser.add_argument(
        "-o",
        "--output-fp",
        default=False,
        help="the output file path which records the analysis results",
    )
    args = parser.parse_args()

    ocr = CnOcr(model_name=MODEL_NAMES, model_epoch=20)
    alphabet = ocr._alphabet

    fn_labels_list = read_input_file(args.input_fp)
    out_f = open(args.output_fp, 'w')

    start_idx = 0
    bad_cnt = 0
    while start_idx < len(fn_labels_list):
        batch = fn_labels_list[start_idx : start_idx + args.batch_size]
        batch_img_fns = []
        batch_labels = []
        batch_imgs = []
        for fn, labels in batch:
            batch_labels.append(labels)
            img_fp = os.path.join(args.image_prefix_dir, fn)
            batch_img_fns.append(img_fp)
            img = mx.image.imread(img_fp, 1).asnumpy()
            batch_imgs.append(img)

        batch_preds = ocr.ocr_for_single_lines(batch_imgs)
        for bad_info in compare_preds_to_reals(
            batch_preds, batch_labels, batch_img_fns, alphabet
        ):
            if args.verbose:
                print(bad_info)
            out_f.write(bad_info + '\n')
            bad_cnt += 1

        start_idx += args.batch_size

    out_f.close()
    print(
        "number of total cases: %d, number of bad cases: %d"
        % (len(fn_labels_list), bad_cnt)
    )


def read_input_file(in_fp):
    fn_labels_list = []
    with open(in_fp) as f:
        for line in f:
            fields = line.strip().split()
            fn_labels_list.append((fields[0], fields[1:]))
    return fn_labels_list


def compare_preds_to_reals(batch_preds, batch_reals, batch_img_fns, alphabet):
    for preds, reals, img_fn in zip(batch_preds, batch_reals, batch_img_fns):
        reals = [alphabet[int(_id)] for _id in reals if _id != '0']  # '0' is padding id
        if preds == reals:
            continue
        preds_set, reals_set = set(preds), set(reals)

        miss_words = reals_set.difference(preds_set)
        redundant_words = preds_set.difference(reals_set)
        yield '%s; real words: %s; pred words: %s; miss words: %s; redundant words: %s' % (
            img_fn,
            ''.join(reals),
            ''.join(preds),
            ''.join(miss_words),
            ''.join(redundant_words),
        )


if __name__ == '__main__':
    evaluate()
