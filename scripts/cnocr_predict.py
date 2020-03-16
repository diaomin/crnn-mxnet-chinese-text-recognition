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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnocr import CnOcr
from cnocr.consts import MODEL_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="model name",
        choices=MODEL_NAMES,
        type=str,
        default='conv-rnn',
    )
    parser.add_argument("-f", "--file", help="Path to the image file")
    parser.add_argument("-s", "--single-line", default=False,
                        help="Whether the image only includes one-line characters")
    args = parser.parse_args()

    ocr = CnOcr(model_name=MODEL_NAMES)
    if args.single_line:
        res = ocr.ocr_for_single_line(args.file)
    else:
        res = ocr.ocr(args.file)
    print("Predicted Chars:", res)


if __name__ == '__main__':
    main()
