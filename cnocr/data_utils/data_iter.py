from __future__ import print_function

import os
from PIL import Image
import numpy as np
from mxnet import nd
import mxnet as mx
import random

from mxnet.image import ImageIter
from mxnet.io import io

from ..utils import normalize_img_array
from .multiproc_data import MPData


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names=list(), label=list()):
        self._data = data
        self._label = label
        self._data_names = data_names
        self._label_names = label_names

        self.pad = 0
        self.index = None  # TODO: what is index?

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def data_names(self):
        return self._data_names

    @property
    def label_names(self):
        return self._label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self._data_names, self._data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self._label_names, self._label)]


# class ImageIter(mx.io.DataIter):
#
#     """
#     Iterator class for generating captcha image data
#     """
#     def __init__(self, data_root, data_list, batch_size, data_shape, num_label, name=None):
#         """
#         Parameters
#         ----------
#         data_root: str
#             root directory of images
#         data_list: str
#             a .txt file stores the image name and corresponding labels for each line
#         batch_size: int
#         name: str
#         """
#         super(ImageIter, self).__init__()
#         self.batch_size = batch_size
#         self.data_shape = data_shape
#         self.num_label  = num_label
#
#         self.data_root = data_root
#         self.dataset_lst_file = open(data_list)
#
#         self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))]
#         self.provide_label = [('label', (self.batch_size, self.num_label))]
#         self.name = name
#
#     def __iter__(self):
#         data = []
#         label = []
#         cnt = 0
#         for m_line in self.dataset_lst_file:
#             img_lst = m_line.strip().split(' ')
#             img_path = os.path.join(self.data_root, img_lst[0])
#
#             cnt += 1
#             img = Image.open(img_path).resize(self.data_shape, Image.BILINEAR).convert('L')
#             img = np.array(img).reshape((1, self.data_shape[1], self.data_shape[0]))
#             data.append(img)
#
#             ret = np.zeros(self.num_label, int)
#             for idx in range(1, len(img_lst)):
#                 ret[idx-1] = int(img_lst[idx])
#
#             label.append(ret)
#             if cnt % self.batch_size == 0:
#                 data_all = [mx.nd.array(data)]
#                 label_all = [mx.nd.array(label)]
#                 data_names = ['data']
#                 label_names = ['label']
#                 data.clear()
#                 label.clear()
#                 yield SimpleBatch(data_names, data_all, label_names, label_all)
#                 continue
#
#
#     def reset(self):
#         if self.dataset_lst_file.seekable():
#             self.dataset_lst_file.seek(0)

class ImageIterLstm(mx.io.DataIter):

    """
    Iterator class for generating captcha image data
    """

    def __init__(self, data_root, data_list, batch_size, data_shape, num_label, lstm_init_states, name=None):
        """
        Parameters
        ----------
        data_root: str
            root directory of images
        data_list: str
            a .txt file stores the image name and corresponding labels for each line
        batch_size: int
        name: str
        """
        super(ImageIterLstm, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label

        self.init_states = lstm_init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in lstm_init_states]

        self.data_root = data_root
        self.dataset_lines = open(data_list).readlines()

        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))] + lstm_init_states
        self.provide_label = [('label', (self.batch_size, self.num_label))]
        self.name = name

    def __iter__(self):
        # init_state_names = [x[0] for x in self.init_states]
        data = []
        label = []
        cnt = 0
        for m_line in self.dataset_lines:
            img_lst = m_line.strip().split(' ')
            img_path = os.path.join(self.data_root, img_lst[0])

            cnt += 1
            img = Image.open(img_path).resize(self.data_shape, Image.BILINEAR).convert('L')
            img = np.array(img).reshape((1, self.data_shape[1], self.data_shape[0]))  # res: [1, height, width]
            data.append(img)

            ret = np.zeros(self.num_label, int)
            for idx in range(1, len(img_lst)):
                ret[idx - 1] = int(img_lst[idx])

            label.append(ret)
            if cnt % self.batch_size == 0:
                data_all = [mx.nd.array(data)]
                label_all = [mx.nd.array(label)]
                data_names = ['data']
                label_names = ['label']
                data = []
                label = []
                yield SimpleBatch(data_names, data_all, label_names, label_all)
                continue

    def reset(self):
        # if self.dataset_lst_file.seekable():
        #     self.dataset_lst_file.seek(0)
        random.shuffle(self.dataset_lines)


class MPOcrImages(object):
    """
    Handles multi-process Chinese OCR image generation
    """
    def __init__(self, data_root, data_list, data_shape, num_label, num_processes, max_queue_size):
        """

        Parameters
        ----------
        data_shape: [width, height]
        num_processes: int
            Number of processes to spawn
        max_queue_size: int
            Maximum images in queue before processes wait
        """
        self.data_shape = data_shape
        self.num_label = num_label

        self.data_root = data_root
        self.dataset_lines = open(data_list).readlines()
        self.total_size = len(self.dataset_lines)
        self.cur_proc_idxs = list(range(num_processes))
        self.num_proc = num_processes

        self.mp_data = MPData(num_processes, max_queue_size, self._gen_sample)

    def _gen_sample(self, proc_id):
        # m_line = random.choice(self.dataset_lines)
        cur_idx = self.cur_proc_idxs[proc_id]
        m_line = self.dataset_lines[cur_idx]
        img_lst = m_line.strip().split(' ')
        img_path = os.path.join(self.data_root, img_lst[0])

        img = Image.open(img_path).resize(self.data_shape, Image.BILINEAR).convert('L')
        img = np.array(img)
        # print(img.shape)
        img = np.transpose(img, (1, 0))  # res: [width, height]
        img = normalize_img_array(img)
        # print(np.mean(img), np.std(img))
        # if len(img.shape) == 2:
        #     img = np.expand_dims(np.transpose(img, (1, 0)), axis=0)  # res: [1, width, height]

        labels = np.zeros(self.num_label, int)
        for idx in range(1, len(img_lst)):
            labels[idx - 1] = int(img_lst[idx])

        self.cur_proc_idxs[proc_id] += self.num_proc
        if self.cur_proc_idxs[proc_id] >= self.total_size:
            self.cur_proc_idxs[proc_id] -= self.total_size

        return img, labels

    @property
    def size(self):
        return len(self.dataset_lines)

    @property
    def shape(self):
        return self.data_shape

    def start(self):
        """
        Starts the processes
        """
        self.mp_data.start()

    def get(self):
        """
        Get an image from the queue

        Returns
        -------
        np.ndarray
            A captcha image, normalized to [0, 1]
        """
        return self.mp_data.get()

    def reset(self):
        """
        Resets the generator by stopping all processes
        """
        self.mp_data.reset()


class OCRIter(mx.io.DataIter):
    """
    Iterator class for generating captcha image data
    """
    def __init__(self, count, batch_size, captcha, num_label, name):
        """
        Parameters
        ----------
        count: int
            Number of batches to produce for one epoch
        batch_size: int
        lstm_init_states: list of tuple(str, tuple)
            A list of tuples with [0] name and [1] shape of each LSTM init state
        captcha MPCaptcha
            Captcha image generator. Can be MPCaptcha or any other class providing .shape and .get() interface
        name: str
        """
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.count = count if count > 0 else captcha.size // batch_size
        # self.init_states = lstm_init_states
        # self.init_state_arrays = [mx.nd.zeros(x[1]) for x in lstm_init_states]
        data_shape = captcha.shape
        # self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))] + lstm_init_states
        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))]
        self.provide_label = [('label', (self.batch_size, num_label))]
        self.mp_captcha = captcha
        self.name = name

    def __iter__(self):
        # init_state_names = [x[0] for x in self.init_states]
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                img, labels = self.mp_captcha.get()
                # print(img.shape)
                img = np.expand_dims(np.transpose(img, (1, 0)), axis=0)  # size: [1, height, width]
                # print(img.shape)
                data.append(img)
                # print('labels', type(labels), labels)
                label.append(labels)
            # data_all = [mx.nd.array(data)] + self.init_state_arrays
            data_all = [mx.nd.array(data)]
            # print(data_all[0].shape)
            label_all = [mx.nd.array(label)]
            # print(label_all[0])
            # data_names = ['data'] + init_state_names
            data_names = ['data']
            label_names = ['label']

            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch


class GrayImageIter(ImageIter):
    def __init__(self, batch_size, data_shape, **kwargs):
        assert 'data_name' not in kwargs and 'label_name' not in kwargs
        super(GrayImageIter, self).__init__(batch_size, data_shape, data_name='data', label_name='label', **kwargs)
        self.provide_data = [('data', (batch_size, 1) + data_shape[1:])]

    def next(self):
        """

        :return: io.DataBatch, which attribute `data` is nd.NDArray,
            with shape [batch_size, 1, height, width] and dtype 'uint8'.
        """
        data_batch = super().next()
        new_data = [self._post_process(sub_data) for sub_data in data_batch.data]

        # data_names = ['data']
        # label_names = ['label']
        # return SimpleBatch(data_names, [new_data], label_names, data_batch.label)
        return io.DataBatch(new_data, data_batch.label, pad=data_batch.pad)

    @classmethod
    def _post_process(cls, data):
        """

        :param data: nd.NDArray with shape [batch_size, channel, height, width]. channel should be 3.
        :return: nd.NDArray with shape [batch_size, 1, height, width] and dtype 'uint8'.
        :param data:
        :return:
        """
        data_shape = list(data.shape)
        data_shape[1] = 1  # [batch_size, 1, height, width]
        new_data = nd.zeros(tuple(data_shape), dtype='float32')

        batch_size = data.shape[0]
        for i in range(batch_size):
            img = data[i]  # shape: [channel, height, width]
            if img.dtype != np.uint8:
                img = img.astype('uint8')
            # color to gray
            img = np.array(Image.fromarray(img.transpose((1, 2, 0)).asnumpy()).convert('L'))
            img = normalize_img_array(img, dtype='float32')
            new_data[i] = nd.expand_dims(nd.array(img), 0)  # res shape: [1, height, width]
        return new_data
