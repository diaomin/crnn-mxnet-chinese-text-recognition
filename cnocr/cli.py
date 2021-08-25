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

from __future__ import absolute_import, division, print_function
import os
import logging
import time
import click
import json
import glob

from torchvision import transforms

from cnocr.consts import MODEL_VERSION, ENCODER_CONFIGS, DECODER_CONFIGS
from cnocr.utils import set_logger, load_model_params, check_model_name
from cnocr.data_utils.aug import NormalizeAug, RandomPaddingAug
from cnocr.dataset import OcrDataModule
from cnocr.trainer import PlTrainer, resave_model
from cnocr import CnOcr, gen_model

_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}
logger = set_logger(log_level=logging.INFO)

DEFAULT_MODEL_NAME = 'densenet-s-fc'
LEGAL_MODEL_NAMES = {
    enc_name + '-' + dec_name
    for enc_name in ENCODER_CONFIGS.keys()
    for dec_name in DECODER_CONFIGS.keys()
}


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('train')
@click.option(
    '-m',
    '--model-name',
    type=click.Choice(LEGAL_MODEL_NAMES),
    default=DEFAULT_MODEL_NAME,
    help='模型名称。默认值为 %s' % DEFAULT_MODEL_NAME,
)
@click.option(
    '-i',
    '--index-dir',
    type=str,
    required=True,
    help='索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件',
)
@click.option(
    '--train-config-fp',
    type=str,
    required=True,
    help='训练使用的json配置文件，参考 `example/train_config.json`',
)
@click.option(
    '-r',
    '--resume-from-checkpoint',
    type=str,
    default=None,
    help='恢复此前中断的训练状态，继续训练。默认为 `None`',
)
@click.option(
    '-p',
    '--pretrained-model-fp',
    type=str,
    default=None,
    help='导入的训练好的模型，作为初始模型。'
    '优先级低于"--restore-training-fp"，当传入"--restore-training-fp"时，此传入失效。默认为 `None`',
)
def train(
    model_name, index_dir, train_config_fp, resume_from_checkpoint, pretrained_model_fp
):
    check_model_name(model_name)
    train_transform = transforms.Compose(
        [
            transforms.RandomInvert(p=0.5),
            # transforms.RandomErasing(p=0.05, scale=(0.01, 0.05)),
            transforms.RandomRotation(degrees=2),
            transforms.RandomAutocontrast(p=0.05),
            NormalizeAug(),
            RandomPaddingAug(p=0.5, max_pad_len=72),
        ]
    )
    val_transform = NormalizeAug()

    train_config = json.load(open(train_config_fp))

    data_mod = OcrDataModule(
        index_dir=index_dir,
        vocab_fp=train_config['vocab_fp'],
        img_folder=train_config['img_folder'],
        train_transforms=train_transform,
        val_transforms=val_transform,
        batch_size=train_config['batch_size'],
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory'],
    )

    trainer = PlTrainer(
        train_config, ckpt_fn=['cnocr', 'v%s' % MODEL_VERSION, model_name]
    )
    model = gen_model(model_name, data_mod.vocab)
    logger.info(model)

    if pretrained_model_fp is not None:
        load_model_params(model, pretrained_model_fp)

    trainer.fit(
        model, datamodule=data_mod, resume_from_checkpoint=resume_from_checkpoint
    )


@cli.command('predict')
@click.option(
    '-m',
    '--model-name',
    type=click.Choice(LEGAL_MODEL_NAMES),
    default=DEFAULT_MODEL_NAME,
    help='模型名称。默认值为 %s' % DEFAULT_MODEL_NAME,
)
@click.option(
    "--model_epoch",
    type=int,
    default=None,
    help="model epoch。默认为 `None`，表示使用系统自带的预训练模型",
)
@click.option(
    '-p',
    '--pretrained-model-fp',
    type=str,
    default=None,
    help='使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型',
)
@click.option(
    "--context",
    help="使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为 `cpu`",
    type=str,
    default='cpu',
)
@click.option("-i", "--img-file-or-dir", required=True, help="输入图片的文件路径或者指定的文件夹")
@click.option(
    "-s",
    "--single-line",
    is_flag=True,
    help="是否输入图片只包含单行文字。对包含单行文字的图片，不做按行切分；否则会先对图片按行分割后再进行识别",
)
def predict(
    model_name, model_epoch, pretrained_model_fp, context, img_file_or_dir, single_line
):
    ocr = CnOcr(
        model_name=model_name,
        model_epoch=model_epoch,
        model_fp=pretrained_model_fp,
        context=context,
    )
    ocr_func = ocr.ocr_for_single_line if single_line else ocr.ocr
    fp_list = []
    if os.path.isfile(img_file_or_dir):
        fp_list.append(img_file_or_dir)
    elif os.path.isdir(img_file_or_dir):
        fn_list = glob.glob1(img_file_or_dir, '*g')
        fp_list = [os.path.join(img_file_or_dir, fn) for fn in fn_list]

    for fp in fp_list:
        start_time = time.time()
        logger.info('\n' + '=' * 10 + fp + '=' * 10)
        res = ocr_func(fp)
        logger.info('time cost: %f' % (time.time() - start_time))
        logger.info(res)
        if single_line:
            res = [res]
        for line_res in res:
            preds, prob = line_res
            logger.info('\npred: %s, with probability %f' % (''.join(preds), prob))


@cli.command('resave')
@click.option('-i', '--input-model-fp', type=str, required=True, help='输入的模型文件路径')
@click.option('-o', '--output-model-fp', type=str, required=True, help='输出的模型文件路径')
def resave_model_file(
    input_model_fp, output_model_fp,
):
    """训练好的模型会存储训练状态，使用此命令去掉预测时无关的数据，降低模型大小"""
    resave_model(input_model_fp, output_model_fp, map_location='cpu')


if __name__ == "__main__":
    cli()
