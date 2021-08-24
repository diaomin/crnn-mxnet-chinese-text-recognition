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

from pathlib import Path
import os
import re
import unicodedata

from cnocr.consts import VOCAB_FP
from cnocr.utils import read_charset, read_tsv_file, set_logger


logger = set_logger(log_level='INFO')


def check_legal(width, height, fname, gt, letter2id):
    legal_letters = set(letter2id)
    if width <= height:
        return False
    if len(gt) < 3 or len(gt) > 30:
        return False
    if gt[0] == '#' and len(gt) == 3 and len(set(gt)) == 1:
        return False

    gt_set = set(gt)
    illegal_letters = gt_set - legal_letters
    if illegal_letters:
        logger.warning(f'illegal letters in gt "{gt}": {illegal_letters}')
        return False

    return True


def transform_gt(gt):
    gt = re.sub(' +', ' ', stringQ2B(gt))
    gt = list(gt)
    return [char if char != ' ' else '<space>' for char in gt]


def read_index_file(fp, letter2id):
    results = []
    with open(fp) as f:
        for line in f:
            width, height, fname, gt = line.strip().split('\t')
            width, height = int(width), int(height)
            gt = transform_gt(gt)
            if check_legal(width, height, fname, gt, letter2id):
                results.append((fname, gt))
    return results


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return unicodedata.normalize('NFKC', ustring)


def process_baidu_innovation():
    vocab, letter2id = read_charset(VOCAB_FP)
    baidu_index_fp = (
        '/Users/king/Documents/beiye-Ein/语料/text-detection/baidu-innovation/train.list'
    )
    out_index_fp = 'train_baidu_innovation.tsv'
    image_folder = 'baidu-innovation/train_images'
    data = read_index_file(baidu_index_fp, letter2id)
    logger.info(f'{len(data)} legal examples are found')

    with open(out_index_fp, 'w') as f:
        for example in data:
            fname, gt = example
            gt = ' '.join(gt)
            fp = image_folder + '/' + fname
            f.write('\t'.join((fp, gt)) + '\n')


def read_csv(fp, letter2id):
    import csv

    results = []
    with open(fp) as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            line_count += 1
            if line_count == 1:
                continue

            fname, gt = row[0], row[1]
            gt = transform_gt(gt)
            if check_legal(280, 32, fname, gt, letter2id):
                results.append((fname, gt))

    return results


def process_baidu_basic():
    vocab, letter2id = read_charset(VOCAB_FP)
    baidu_index_fp = (
        '/Users/king/Documents/beiye-Ein/语料/text-detection/baidu-basic/train_label.csv'
    )
    out_index_fp = 'train_baidu_basic.tsv'
    image_folder = 'baidu-basic/train_images'
    data = read_csv(baidu_index_fp, letter2id)

    logger.info(f'{len(data)} legal examples are found')

    with open(out_index_fp, 'w') as f:
        for example in data:
            fname, gt = example
            gt = ' '.join(gt)
            fp = image_folder + '/' + fname
            f.write('\t'.join((fp, gt)) + '\n')


if __name__ == '__main__':
    # process_baidu_innovation()
    process_baidu_basic()
