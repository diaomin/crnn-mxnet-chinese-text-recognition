import os
from copy import deepcopy
import mxnet as mx
import numpy as np

from cnocr.consts import MODEL_EPOCE
from cnocr.hyperparams.cn_hyperparams import CnHyperparams as Hyperparams
from cnocr.fit.lstm import init_states
from cnocr.fit.ctc_metrics import CtcMetrics
from cnocr.data_utils.data_iter import SimpleBatch
from cnocr.symbols.crnn import crnn_lstm
from cnocr.utils import data_dir, get_model_file, read_charset


def read_ocr_img(path):
    """
    :param path: image file path
    :return: gray image, with dim [height, width, 1]
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
    MODEL_FILE_PREFIX = 'model'

    def __init__(self, root=data_dir(), model_epoch=MODEL_EPOCE):
        self._model_dir = os.path.join(root, 'models')
        self._model_epoch = model_epoch
        self._assert_and_prepare_model_files(root)
        self._alphabet, _ = read_charset(os.path.join(self._model_dir, 'label_cn.txt'))

        self._hp = Hyperparams()
        self._mods = {}

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

    def _get_module(self, hp, sample):
        network = crnn_lstm(hp)
        prefix = os.path.join(self._model_dir, self.MODEL_FILE_PREFIX)
        mod = load_module(prefix, MODEL_EPOCE, sample.data_names, sample.provide_data, network=network)
        return mod

    def ocr_for_single_line(self, img_fp):
        """
        Recognize characters from an image with characters with only one line
        :param img_fp: image file path; or gray image mx.nd.NDArray; or gray image np.ndarray
        :return: charector list, such as ['你', '好']
        """
        hp = deepcopy(self._hp)
        if isinstance(img_fp, str) and os.path.isfile(img_fp):
            img = read_ocr_img(img_fp)
        elif isinstance(img_fp, mx.nd.NDArray) or isinstance(img_fp, np.ndarray):
            img = img_fp
        else:
            raise TypeError('Inappropriate argument type.')
        img = rescale_img(img, hp)

        init_state_names, init_state_arrays = lstm_init_states(batch_size=1, hp=hp)

        sample = SimpleBatch(
            data_names=['data'] + init_state_names,
            data=[mx.nd.array([img])] + init_state_arrays)

        mod = self._get_module(hp, sample)

        mod.forward(sample)
        prob = mod.get_outputs()[0].asnumpy()

        prediction, start_end_idx = CtcMetrics.ctc_label(np.argmax(prob, axis=-1).tolist())
        # print(start_end_idx)

        alphabet = self._alphabet
        res = [alphabet[p] for p in prediction]
        return res
