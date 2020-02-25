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
import mxnet as mx
import numpy as np
from PIL import Image

from cnocr.__version__ import __version__
from cnocr.consts import MODEL_EPOCE
from cnocr.hyperparams.cn_hyperparams import CnHyperparams as Hyperparams
from cnocr.fit.lstm import init_states
from cnocr.fit.ctc_metrics import CtcMetrics
from cnocr.data_utils.data_iter import SimpleBatch
from cnocr.symbols.crnn import crnn_lstm
from cnocr.utils import data_dir, get_model_file, read_charset, normalize_img_array
from cnocr.line_split import line_split


def read_ocr_img(path):
    """
    :param path: image file path
    :return: gray image, with dim [height, width, 1], with values range from 0 to 255
    """
    # img = Image.open(path).resize((hp.img_width, hp.img_height), Image.BILINEAR)
    # img = img.convert('L')
    # img = np.expand_dims(np.array(img), 0)
    # return img
    return mx.image.imread(path, 0)


def rescale_img(img, hp):
    """

    :param img: np.ndarray or mx.ndarray; should be gray image, with dim [height, width] or [height, width, 1]
    :param hp: instance of Hyperparams
    :return: np.ndarray with the given width and height from hp. The resulting dim is [1, height, width]
    """
    if isinstance(img, np.ndarray):
        img = mx.nd.array(img)
    scale = hp.img_height / img.shape[0]
    new_width = int(scale * img.shape[1])
    hp._seq_length = new_width // 8
    if len(img.shape) == 2:  # mx.image.imresize needs the third dim
        img = mx.nd.expand_dims(img, 2)
    img = mx.image.imresize(img, w=new_width, h=hp.img_height).asnumpy()
    img = np.squeeze(img, axis=2)
    return np.expand_dims(img, 0)


def lstm_init_states(batch_size, hp):
    """ Returns a tuple of names and zero arrays for LSTM init states"""
    init_shapes = init_states(batch_size=batch_size, num_lstm_layer=hp.num_lstm_layer, num_hidden=hp.num_hidden)
    init_names = [s[0] for s in init_shapes]
    init_arrays = [mx.nd.zeros(x[1]) for x in init_shapes]
    # init_names.append('seq_length')
    # init_arrays.append(hp.seq_length)
    return init_names, init_arrays


def load_module(prefix, epoch, data_names, data_shapes, network=None):
    """
    Loads the model from checkpoint specified by prefix and epoch, binds it
    to an executor, and sets its parameters and returns a mx.mod.Module
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    if network is not None:
        sym = network

    # We don't need CTC loss for prediction, just a simple softmax will suffice.
    # We get the output of the layer just before the loss layer ('pred_fc') and add softmax on top
    pred_fc = sym.get_internals()['pred_fc_output']
    sym = mx.sym.softmax(data=pred_fc)

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=data_names, label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=False)
    return mod


class CnOcr(object):
    MODEL_FILE_PREFIX = 'model-v{}'.format(__version__)

    def __init__(self, root=data_dir(), model_epoch=MODEL_EPOCE):
        self._model_dir = os.path.join(root, 'models')
        self._model_epoch = model_epoch
        self._assert_and_prepare_model_files(root)
        self._alphabet, _ = read_charset(os.path.join(self._model_dir, 'label_cn.txt'))

        self._hp = Hyperparams()
        self._hp._loss_type = None  # infer mode

        self._mod = self._get_module(self._hp)

    def _assert_and_prepare_model_files(self, root):
        model_dir = self._model_dir
        model_files = ['label_cn.txt',
                       '%s-%04d.params' % (self.MODEL_FILE_PREFIX, self._model_epoch),
                       '%s-symbol.json' % self.MODEL_FILE_PREFIX]
        file_prepared = True
        for f in model_files:
            f = os.path.join(model_dir, f)
            if not os.path.exists(f):
                file_prepared = False
                break

        if file_prepared:
            return

        if os.path.exists(model_dir):
            os.removedirs(model_dir)
        get_model_file(root)

    def _get_module(self, hp):
        network = crnn_lstm(hp)
        prefix = os.path.join(self._model_dir, self.MODEL_FILE_PREFIX)
        # import pdb; pdb.set_trace()
        data_names = ['data']
        data_shapes = [(data_names[0], (hp.batch_size, 1, hp.img_height, hp.img_width))]
        mod = load_module(prefix, self._model_epoch, data_names, data_shapes, network=network)
        return mod

    def ocr(self, img_fp):
        """
        :param img_fp: image file path; or color image mx.nd.NDArray or np.ndarray,
            with shape (height, width, 3), and the channels should be RGB formatted.
        :return: List(List(Char)), such as:
            [['第', '一', '行'], ['第', '二', '行'], ['第', '三', '行']]
        """
        if isinstance(img_fp, str) and os.path.isfile(img_fp):
            img = mx.image.imread(img_fp, 1).asnumpy()
        elif isinstance(img_fp, mx.nd.NDArray):
            img = img_fp.asnumpy()
        elif isinstance(img_fp, np.ndarray):
            img = img_fp
        else:
            raise TypeError('Inappropriate argument type.')
        if min(img.shape[0], img.shape[1]) < 2:
            return ''
        line_imgs = line_split(img, blank=True)
        line_img_list = [line_img for line_img, _ in line_imgs]
        line_chars_list = self.ocr_for_single_lines(line_img_list)
        return line_chars_list

    def ocr_for_single_line(self, img_fp):
        """
        Recognize characters from an image with only one-line characters.
        :param img_fp: image file path; or image mx.nd.NDArray or np.ndarray,
            with shape [height, width] or [height, width, channel].
            The optional channel should be 1 (gray image) or 3 (color image).
        :return: character list, such as ['你', '好']
        """
        if isinstance(img_fp, str) and os.path.isfile(img_fp):
            img = read_ocr_img(img_fp)
        elif isinstance(img_fp, mx.nd.NDArray) or isinstance(img_fp, np.ndarray):
            img = img_fp
        else:
            raise TypeError('Inappropriate argument type.')
        res = self.ocr_for_single_lines([img])
        return res[0]

    def ocr_for_single_lines(self, img_list):
        """
        Batch recognize characters from a list of one-line-characters images.
        :param img_list: list of images, in which each element should be a line image array,
            with type mx.nd.NDArray or np.ndarray.
            Each element should be a tensor with values ranging from 0 to 255,
            and with shape [height, width] or [height, width, channel].
            The optional channel should be 1 (gray image) or 3 (color image).
        :return: list of list of chars, such as
            [['第', '一', '行'], ['第', '二', '行'], ['第', '三', '行']]
        """
        if len(img_list) == 0:
            return []
        img_list = [self._preprocess_img_array(img) for img in img_list]

        batch_size = len(img_list)
        img_list, img_widths = self._pad_arrays(img_list)

        # import pdb; pdb.set_trace()
        sample = SimpleBatch(
            data_names=['data'],
            data=[mx.nd.array(img_list)])

        prob = self._predict(sample)
        prob = np.reshape(prob, (-1, batch_size, prob.shape[1]))  # [seq_len, batch_size, num_classes]

        max_width = max(img_widths)
        res = []
        for i in range(batch_size):
            res.append(self._gen_line_pred_chars(prob[:, i, :], img_widths[i], max_width))
        return res

    def _preprocess_img_array(self, img):
        """
        :param img: image array with type mx.nd.NDArray or np.ndarray,
        with shape [height, width] or [height, width, channel].
        channel shoule be 1 (gray image) or 3 (color image).

        :return: np.ndarray, with shape (1, height, width)
        """
        if len(img.shape) == 3 and img.shape[2] == 3:
            if isinstance(img, mx.nd.NDArray):
                img = img.asnumpy()
            if img.dtype != np.dtype('uint8'):
                img = img.astype('uint8')
            # color to gray
            img = np.array(Image.fromarray(img).convert('L'))
        img = rescale_img(img, self._hp)
        return normalize_img_array(img)

    def _pad_arrays(self, img_list):
        """Padding to make sure all the elements have the same width."""
        img_widths = [img.shape[2] for img in img_list]
        if len(img_list) <= 1:
            return img_list, img_widths
        max_width = max(img_widths)
        pad_width = [(0, 0), (0, 0), (0, 0)]
        padded_img_list = []
        for img in img_list:
            if img.shape[2] < max_width:
                pad_width[2] = (0, max_width - img.shape[2])
                img = np.pad(img, pad_width, 'constant', constant_values=0.0)
            padded_img_list.append(img)
        return padded_img_list, img_widths

    def _predict(self, sample):
        mod = self._mod
        mod.forward(sample)
        prob = mod.get_outputs()[0].asnumpy()
        return prob

    def _gen_line_pred_chars(self, line_prob, img_width, max_img_width):
        """
        Get the predicted characters.
        :param line_prob: with shape of [seq_length, num_classes]
        :param img_width:
        :param max_img_width:
        :return:
        """
        class_ids = np.argmax(line_prob, axis=-1)
        # idxs = list(zip(range(len(class_ids)), class_ids))
        # probs = [line_prob[e[0], e[1]] for e in idxs]

        if img_width < max_img_width:
            comp_ratio = self._hp.seq_len_cmpr_ratio
            end_idx = img_width // comp_ratio
            if end_idx < len(class_ids):
                class_ids[end_idx:] = 0
        prediction, start_end_idx = CtcMetrics.ctc_label(class_ids.tolist())
        # print(start_end_idx)
        alphabet = self._alphabet
        res = [alphabet[p] if alphabet[p] != '<space>' else ' ' for p in prediction]

        # res = self._insert_space_char(res, start_end_idx)
        return res

    def _insert_space_char(self, pred_chars, start_end_idx, min_interval=None):
        if len(pred_chars) < 2:
            return pred_chars
        assert len(pred_chars) == len(start_end_idx)

        if min_interval is None:
            # 自动计算最小区间值
            intervals = {start_end_idx[idx][0] - start_end_idx[idx-1][1] for idx in range(1, len(start_end_idx))}
            if len(intervals) >= 3:
                intervals = sorted(list(intervals))
                if intervals[0] < 1:  # 排除间距为0的情况
                    intervals = intervals[1:]
                min_interval = intervals[2]
            else:
                min_interval = start_end_idx[-1][1]  # no space will be inserted

        res_chars = [pred_chars[0]]
        for idx in range(1, len(pred_chars)):
            if start_end_idx[idx][0] - start_end_idx[idx-1][1] >= min_interval:
                res_chars.append(' ')
            res_chars.append(pred_chars[idx])
        return res_chars
