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
# Credits: adapted from https://mp.weixin.qq.com/s/xGvaW87UQFjetc5xFmKxWg


import random
from torch.utils.data import Dataset, DataLoader


class BlockShuffleDataLoader(DataLoader):
    def __init__(
        self, dataset: Dataset, **kwargs
    ):
        """
        对 OcrDataset 数据集实现Block Shuffle功能，按文字数量从少到多的顺序排列样本（相同长度样本则随机排列）
        Args:
            dataset: OcrDataset类的实例，其中中必须包含labels_list变量，并且该变量为一个list
            **kwargs:
        """
        assert isinstance(
            dataset.labels_list, list
        ), "dataset为OcrDataset类的实例，其中必须包含labels_list变量，并且该变量为一个list"
        kwargs['shuffle'] = False
        super().__init__(dataset, **kwargs)

    def __iter__(self):
        self.block_shuffle2()
        return super().__iter__()

    def block_shuffle2(self):
        idx_list = list(range(len(self.dataset)))
        random.shuffle(idx_list)
        random.shuffle(idx_list)
        idx_list.sort(key=lambda idx: len(self.dataset.labels_list[idx]))
        for attr in ('img_fp_list', 'labels_list'):
            ori_list = getattr(self.dataset, attr)
            new_list = [ori_list[idx] for idx in idx_list]
            setattr(self.dataset, attr, new_list)
