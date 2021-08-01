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
import os
from pathlib import Path
import logging
import platform
import zipfile
from PIL import Image
from typing import Union, Any, Tuple

import numpy as np
import torch
from torch import Tensor
import mxnet as mx
from mxnet.gluon.utils import download
from torchvision.utils import save_image

from .consts import AVAILABLE_MODELS, EMB_MODEL_TYPES, SEQ_MODEL_TYPES


fmt = '[%(levelname)s %(asctime)s %(funcName)s:%(lineno)d] %(' \
      'message)s '
logging.basicConfig(format=fmt)
logging.captureWarnings(True)
logger = logging.getLogger()


def set_logger(log_file=None, log_level=logging.INFO, log_file_level=logging.NOTSET):
    """
    Example:
        >>> set_logger(log_file)
        >>> logger.info("abc'")
    """
    log_format = logging.Formatter(fmt)
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        if not Path(log_file).parent.exists():
            os.makedirs(Path(log_file).parent)
        if isinstance(log_file, Path):
            log_file = str(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def gen_context(num_gpu):
    if num_gpu > 0:
        context = [mx.context.gpu(i) for i in range(num_gpu)]
    else:
        context = [mx.context.cpu()]
    return context


def check_context(context):
    if isinstance(context, str):
        return context.lower() in ('gpu', 'cpu')
    if isinstance(context, list):
        if len(context) < 1:
            return False
        return all(isinstance(ctx, mx.Context) for ctx in context)
    return isinstance(context, mx.Context)


def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'cnocr')
    else:
        return os.path.join(os.path.expanduser("~"), '.cnocr')


def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('CNOCR_HOME', data_dir_default())


def check_model_name(model_name):
    emb_model_type, seq_model_type = model_name.rsplit('-', maxsplit=1)
    assert emb_model_type in EMB_MODEL_TYPES
    assert seq_model_type in SEQ_MODEL_TYPES


def get_model_file(model_dir):
    r"""Return location for the downloaded models on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_dir : str, default $CNOCR_HOME
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    model_dir = os.path.expanduser(model_dir)
    par_dir = os.path.dirname(model_dir)
    os.makedirs(par_dir, exist_ok=True)

    zip_file_path = model_dir + '.zip'
    if not os.path.exists(zip_file_path):
        model_name = os.path.basename(model_dir)
        if model_name not in AVAILABLE_MODELS:
            raise NotImplementedError('%s is not an available downloaded model' % model_name)
        url = AVAILABLE_MODELS[model_name][1]
        download(url, path=zip_file_path, overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(par_dir)
    os.remove(zip_file_path)

    return model_dir


def read_charset(charset_fp):
    alphabet = []
    with open(charset_fp, encoding='utf-8') as fp:
        for line in fp:
            alphabet.append(line.rstrip('\n'))
    # print('Alphabet size: %d' % len(alphabet))
    # try:
    #     space_idx = alphabet.index('<space>')
    #     alphabet[space_idx] = ' '
    # except ValueError:
    #     pass
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    return alphabet, inv_alph_dict


def read_tsv_file(fp, sep='\t', img_folder=None, mode='eval'):
    img_fp_list, labels_list = [], []
    num_fields = 2 if mode != 'test' else 1
    with open(fp) as f:
        for line in f:
            fields = line.strip('\n').split(sep)
            assert len(fields) == num_fields
            img_fp = os.path.join(img_folder, fields[0]) if img_folder is not None else fields[0]
            img_fp_list.append(img_fp)

            if mode != 'test':
                labels = fields[1].split(' ')
                labels_list.append(labels)

    return (img_fp_list, labels_list) if mode != 'test' else (img_fp_list, None)


def read_img(path):
    """
    :param path: image file path
    :return: gray image, with dim [1, height, width], with values range from 0 to 255
    """
    img = Image.open(path)
    img = img.convert('L')
    img = np.expand_dims(np.array(img), 0)
    return img


def save_img(img: Union[Tensor, np.ndarray], path):
    if not isinstance(img, Tensor):
        img = torch.from_numpy(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    # img *= 255
    # img = img.to(dtype=torch.uint8)
    save_image(img, path)

    # Image.fromarray(img).save(path)


def normalize_img_array(img: Union[Tensor, np.ndarray]):
    """ rescale """
    if isinstance(img, Tensor):
        img = img.to(dtype=torch.float32)
    else:
        img = img.astype('float32')
    # return (img - np.mean(img, dtype=dtype)) / 255.0
    return img / 255.0
    # return (img - np.median(img)) / (np.std(img, dtype=dtype) + 1e-6)  # 转完以后有些情况会变得不可识别


def gen_length_mask(lengths: torch.Tensor, mask_size: Union[Tuple, Any]):
    labels = torch.arange(mask_size[-1], device=lengths.device, dtype=torch.long)
    while True:
        if len(labels.shape) >= len(mask_size):
            break
        labels = labels.unsqueeze(0)
        lengths = lengths.unsqueeze(-1)
    mask = labels < lengths
    return ~mask
