# coding: utf-8
import os
import sys
from pathlib import Path
import mxnet as mx
import numpy as np
from mxnet import nd
import pytest
from mxnet.image import ImageIter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr.data_utils.aug import FgBgFlipAug
from cnocr.data_utils.data_iter import GrayImageIter

LST_DIR = Path('data/lst')
DATA_DIR = Path('data/sample-data')


def test_nd():
    ele = np.reshape(np.array(range(2 * 3)), (2, 3))
    data = [ele, ele + 10]
    new = nd.array([ele])
    assert new.shape == (1, 2, 3)
    new = nd.array(data)
    assert new.shape == (2, 2, 3)
    print(new)


def _read_lst_file(fp):
    with open(fp) as f:
        for line in f:
            _, fname = line.strip().rsplit('\t', maxsplit=1)
            yield str(DATA_DIR / fname)


@pytest.mark.parametrize(
    'fp_prefix',
    [
        LST_DIR / 'sample-data_test',
        # LST_DIR / 'sample-data_0_train',
        # LST_DIR / 'sample-data_1_train',
        # LST_DIR / 'sample-data_2_train',
    ],
)
def test_iter(fp_prefix):
    augs = mx.image.CreateAugmenter(
        data_shape=(3, 32, 280),
        resize=0,
        rand_crop=False,
        rand_resize=False,
        rand_mirror=False,
        mean=None,
        std=None,
        brightness=0.05,
        contrast=0.1,
        saturation=0.3,
        hue=0.1,
        pca_noise=0.3,
        inter_method=2,
    )
    augs.append(FgBgFlipAug(p=0.2))
    data_iter = GrayImageIter(
        batch_size=2,
        data_shape=(3, 32, 280),
        label_width=10,
        path_imgrec=str(fp_prefix) + ".rec",
        path_imgidx=str(fp_prefix) + ".idx",
        aug_list=augs,
    )

    expected_img_fps = _read_lst_file(str(fp_prefix) + ".lst")
    expected_imgs = [
        mx.image.imread(fp, 1) for fp in expected_img_fps
    ]  # shape of each one: (32, 280, 3)

    # data_iter的类型是mxnet.image.ImageIter
    # reset()函数的作用是：resents the iterator to the beginning of the data
    data_iter.reset()

    # batch的类型是mxnet.io.DataBatch，因为next()方法的返回值就是DataBatch
    batch = data_iter.next()

    # data是一个NDArray，表示第一个batch中的数据，因为这里的batch_size大小是4，所以data的size是2*3*32*280
    data = batch.data[0]  # shape of each one: (3, 32, 280)

    from matplotlib import pyplot as plt

    # 这个for循环就是读取这个batch中的每张图像并显示
    for i in range(2):
        plt.subplot(4, 1, i * 2 + 1)
        print(data[i].shape)
        # print(
        #     nd.sum(nd.abs(data[i].astype(np.uint8))),
        #     nd.sum(expected_imgs[i].transpose((2, 0, 1))),
        # )
        # print(
        #     nd.sum(
        #         nd.abs(data[i].astype(np.uint8) - expected_imgs[i].transpose((2, 0, 1)))
        #     )
        # )
        print(float(data[i].min()), float(data[i].max()))
        new_img = data[i].asnumpy() * 255
        plt.imshow(new_img.astype(np.uint8).squeeze(axis=0), cmap='gray')
        plt.subplot(4, 1, i * 2 + 2)
        plt.imshow(expected_imgs[i].asnumpy())
    plt.show()
