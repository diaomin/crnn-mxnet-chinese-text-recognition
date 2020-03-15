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
import platform
import zipfile
import numpy as np
from mxnet.gluon.utils import download

from .consts import MODEL_BASE_URL, ZIP_FILE_NAME


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


def get_model_file(root=data_dir()):
    r"""Return location for the downloaded models on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    root : str, default $CNOCR_HOME
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    root = os.path.expanduser(root)

    os.makedirs(root, exist_ok=True)

    zip_file_path = os.path.join(root, ZIP_FILE_NAME)
    if not os.path.exists(zip_file_path):
        download(MODEL_BASE_URL, path=zip_file_path, overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    return os.path.join(root, 'models')


def read_charset(charset_fp):
    alphabet = [None]
    # 第0个元素是预留id，在CTC中用来分割字符。它不对应有意义的字符
    with open(charset_fp, encoding='utf-8') as fp:
        for line in fp:
            alphabet.append(line.rstrip('\n'))
    # print('Alphabet size: %d' % len(alphabet))
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    # inv_alph_dict[' '] = inv_alph_dict['<space>']  # 对应空格
    return alphabet, inv_alph_dict


def normalize_img_array(img, dtype='float32'):
    """ rescale to [-1.0, 1.0] """
    img = img.astype(dtype)
    # return (img - np.mean(img, dtype=dtype)) / 255.0
    return img / 255.0
    # return (img - np.median(img)) / (np.std(img, dtype=dtype) + 1e-6)  # 转完以后有些情况会变得不可识别
