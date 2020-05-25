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
import glob
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnocr import CnOcr
from cnocr.utils import set_logger


logger = set_logger(log_level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", help="model name", type=str, default='conv-lite-fc'
    )
    parser.add_argument("--model_epoch", type=int, default=None, help="model epoch")
    parser.add_argument("-f", "--file", help="Path to the image file or dir")
    parser.add_argument(
        "-s",
        "--single-line",
        default=False,
        help="Whether the image only includes one-line characters",
    )
    args = parser.parse_args()

    ocr = CnOcr(model_name=args.model_name, model_epoch=args.model_epoch)
    ocr_func = ocr.ocr_for_single_line if args.single_line else ocr.ocr
    fp_list = []
    if os.path.isfile(args.file):
        fp_list.append(args.file)
    elif os.path.isdir(args.file):
        fn_list = glob.glob1(args.file, '*g')
        fp_list = [os.path.join(args.file, fn) for fn in fn_list]

    for fp in fp_list:
        res = ocr_func(fp)
        logger.info('\n' + '='*10 + fp + '='*10)
        res = [''.join(line_p) for line_p in res]
        logger.info('\n' + '\n'.join(res))


if __name__ == '__main__':
    main()
