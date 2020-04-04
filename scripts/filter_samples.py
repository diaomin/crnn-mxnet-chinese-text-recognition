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
从badcases.txt中可以发现一些不好的训练样本。这个脚本就是为了过滤掉这些样本。
"""
from __future__ import print_function

import argparse
import logging


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_file", help="Path to train or test txt file", type=str, required=True
    )
    parser.add_argument(
        "--badcases_file", help="Path to badcases file from evaluate mode", type=str, required=True
    )
    parser.add_argument(
        "--distance_thrsh", help="samples with distance >= thrsh will be deleted", type=int, default=2
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help="the new sample file",
        type=str,
        required=True,
    )
    return parser.parse_args()


def read_badcases_file(fp, dist_thrsh):
    badcases = set()
    with open(fp) as f:
        for line in f:
            line = line.strip()
            fields = line.split('\t')
            if fields[0] == 'distance':
                continue
            dist, fp = int(fields[0]), fields[1]
            if dist >= dist_thrsh:
                fp = '/'.join(fp.split('/')[-2:])
                badcases.add(fp)
    print('get %d badcase samples' % len(badcases))
    return badcases


def process_sample_file(in_fp, out_fp, badcases):
    num_deleted = 0
    with open(in_fp) as in_f, open(out_fp, 'w') as out_f:
        for line in in_f:
            line = line.strip()
            fields = line.split()
            sample_fp = fields[0]

            if sample_fp in badcases:
                num_deleted += 1
            else:
                out_f.write(line + '\n')
    print('%d samples are deleted' % num_deleted)


def filter(args):
    """选择包含给定id的样本，并按包含数量从高到低排序。"""
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    badcases = read_badcases_file(args.badcases_file, args.distance_thrsh)
    process_sample_file(args.sample_file, args.output_file, badcases)


if __name__ == '__main__':
    args = parse_args()
    filter(args)
