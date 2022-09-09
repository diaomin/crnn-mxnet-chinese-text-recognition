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

import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple, Callable

import pytorch_lightning as pt
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils import read_img


logger = logging.getLogger(__name__)


def read_tsv_file(fp, sep='\t', img_folder=None, mode='eval'):
    """
    format of each line:
        <label>\t<ulr>
    """
    img_fp_list, labels_list = [], []
    num_fields = 2 if mode != 'test' else 1
    with open(fp) as f:
        for line in f:
            fields = line.strip('\n').split(sep)
            assert len(fields) == num_fields
            img_fp = (
                os.path.join(img_folder, fields[-1])
                if img_folder is not None
                else fields[-1]
            )
            img_fp_list.append(img_fp)

            if mode != 'test':
                labels = fields[0]
                labels_list.append(labels)

    return (img_fp_list, labels_list) if mode != 'test' else (img_fp_list, None)


class ImageDataset(Dataset):
    def __init__(self, categories, index_fp, img_folder=None, mode='train'):
        super().__init__()

        self.categories = categories
        self.category_dict = {_name: idx for idx, _name in enumerate(self.categories)}

        self.img_fp_list, self.labels_list = read_tsv_file(
            index_fp, '\t', img_folder, mode
        )
        self.mode = mode

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, item):
        img_fp = self.img_fp_list[item]
        try:
            img = torch.tensor(
                read_img(img_fp, gray=False).transpose((2, 0, 1))
            )  # res: [3, H, W]
        except:
            logger.error(f'unsupported file {img_fp}')
            breakpoint()

        if self.mode != 'test':
            labels = self.category_dict[self.labels_list[item]]

        return (img, labels) if self.mode != 'test' else (img,)


def collate_fn(img_labels: List[Tuple[str, str]], transformers: Callable = None):
    test_mode = len(img_labels[0]) == 1
    if test_mode:
        img_list = zip(*img_labels)
        labels = None
    else:
        img_list, labels = zip(*img_labels)

    if transformers is not None:
        new_img_list = []
        new_labels = []
        for idx, img in enumerate(img_list):
            try:
                new_img_list.append(transformers(img))
                if labels is not None:
                    new_labels.append(labels[idx])
            except:
                continue
        img_list = new_img_list
        if labels is not None:
            labels = torch.tensor(new_labels)
    imgs = torch.stack(img_list)
    return imgs, labels


class ImageDataModule(pt.LightningDataModule):
    def __init__(
        self,
        categories: Union[list, tuple],
        index_dir: Union[str, Path],
        img_folder: Union[str, Path, None] = None,
        train_transforms=None,
        val_transforms=None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.index_dir = Path(index_dir)
        self.img_folder = img_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        self.train = ImageDataset(
            categories, self.index_dir / 'train.tsv', self.img_folder, mode='train'
        )
        self.val = ImageDataset(
            categories, self.index_dir / 'dev.tsv', self.img_folder, mode='train'
        )

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
