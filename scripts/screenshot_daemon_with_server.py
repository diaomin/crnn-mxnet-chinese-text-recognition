# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).
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

# 也可以下调用命令在命令行调用开启的OCR服务：
# > curl -F image=@docs/examples/huochepiao.jpeg http://0.0.0.0:8501/ocr

import os
import time
import glob
import requests
from pprint import pformat

import pyperclip as pc
from cnocr.utils import set_logger

logger = set_logger(log_level='DEBUG')


SERVICE_URL = os.getenv("CNOCR_SERVICE", 'http://0.0.0.0:8501/ocr')
SCREENSHOT_DIR = os.getenv(
    "SCREENSHOT_DIR", '/Users/king/Pictures/screenshot_from_xnip'
)


def ocr(image):
    r = requests.post(
        SERVICE_URL, files={'image': (image, open(image, 'rb'), 'image/png')},
    )
    return r.json()


def get_newest_fp_time(screenshot_dir):
    fn_list = glob.glob1(screenshot_dir, '*g')
    fp_list = [os.path.join(screenshot_dir, fn) for fn in fn_list]
    if not fp_list:
        return None, None
    fp_list.sort(key=lambda fp: os.path.getmtime(fp), reverse=True)
    return fp_list[0], os.path.getmtime(fp_list[0])


def ocr_newest(screenshot_dir, delta_interval):
    while True:
        newest_fp, newest_mod_time = get_newest_fp_time(screenshot_dir)
        if (
            newest_mod_time is not None
            and time.time() - newest_mod_time < delta_interval
        ):
            logger.info(f'analyzing screenshot file {newest_fp}')
            result = ocr(newest_fp)
            texts = [_one['text'] for _one in result['results']]
            logger.info(f'\tOCR results: {pformat(texts)}\n\n')
            if texts:
                pc.copy('\n'.join(texts))
        time.sleep(1)


if __name__ == '__main__':
    ocr_newest(SCREENSHOT_DIR, 2)
