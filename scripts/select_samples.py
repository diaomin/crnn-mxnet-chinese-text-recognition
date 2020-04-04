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
"""
选择包含给定id的样本，并按包含数量从高到低排序。
主要是期望找出l相关的样本，然后看看模型能否预测出ll这种在一块的占位很小的序列。
"""
from __future__ import print_function

import argparse
import logging
import os
import shutil


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_file", help="Path to train or test txt file", type=str, required=True
    )
    parser.add_argument(
        "--prefix", help="prefix directory for image files", required=True
    )
    parser.add_argument(
        "--target_id", help="target id to select", type=int, default=242
    )
    parser.add_argument(
        "--num_samples", help="how many samples to be selected", type=int, default=256
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="the directory for storing selected sample images",
        type=str,
        required=True,
    )
    return parser.parse_args()


def read_file(fp, target_id):
    target_id = str(target_id)
    res_list = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            fields = line.split()
            sample_fp, ids = fields[0], fields[1:]
            target_cnt = ids.count(target_id)
            if target_cnt > 0:
                res_list.append((target_cnt, sample_fp, line))
    return res_list


def copy_files(cand_list, prefix, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, 'labels.txt'), 'w') as label_f:
        for _, sample_fp, line in cand_list:
            fp = os.path.join(prefix, sample_fp)
            shutil.copy(fp, out_dir)
            label_f.write(line + '\n')


def select(args):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    cand_list = read_file(args.sample_file, args.target_id)
    cand_list.sort(key=lambda x: x[0], reverse=True)
    cand_list = cand_list[: args.num_samples]
    copy_files(cand_list, args.prefix, args.output_dir)


if __name__ == '__main__':
    args = parse_args()
    select(args)
