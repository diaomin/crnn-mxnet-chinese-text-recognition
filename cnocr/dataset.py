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
from typing import Optional, Union, List, Tuple, Callable

import pytorch_lightning as pt
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import read_charset, read_tsv_file, read_img, rescale_img, pad_img_seq


class OcrDataset(Dataset):
    def __init__(self, index_fp, img_folder=None, mode='train'):
        super().__init__()
        self.img_fp_list, self.labels_list = read_tsv_file(
            index_fp, '\t', img_folder, mode
        )
        self.mode = mode

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, item):
        img_fp = self.img_fp_list[item]
        img = read_img(img_fp).transpose((2, 0, 1))  # res: [1, H, W]
        img = rescale_img(img)

        if self.mode != 'test':
            labels = self.labels_list[item]
            # label_ids = [self.letter2id[l] for l in labels]

        return (img, labels) if self.mode != 'test' else (img,)


def collate_fn(img_labels: List[Tuple[str, str]], transformers: Callable = None):
    test_mode = len(img_labels[0]) == 1
    if test_mode:
        img_list = zip(*img_labels)
        labels_list, label_lengths = None, None
    else:
        img_list, labels_list = zip(*img_labels)
        label_lengths = torch.tensor([len(labels) for labels in labels_list])

    img_lengths = torch.tensor([img.size(2) for img in img_list])
    if transformers is not None:
        img_list = [transformers(img) for img in img_list]
    imgs = pad_img_seq(img_list)
    return imgs, img_lengths, labels_list, label_lengths


class OcrDataModule(pt.LightningDataModule):
    def __init__(
        self,
        index_dir: Union[str, Path],
        vocab_fp: Union[str, Path],
        img_folder: Union[str, Path, None] = None,
        train_transforms=None,
        val_transforms=None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(
            train_transforms=train_transforms, val_transforms=val_transforms
        )
        self.vocab, self.letter2id = read_charset(vocab_fp)
        self.index_dir = Path(index_dir)
        self.img_folder = img_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train = OcrDataset(
            self.index_dir / 'train.tsv', self.img_folder, mode='train'
        )
        self.val = OcrDataset(self.index_dir / 'dev.tsv', self.img_folder, mode='train')

    @property
    def vocab_size(self):
        return len(self.vocab)

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, self.train_transforms),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: collate_fn(x, self.val_transforms),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return None
