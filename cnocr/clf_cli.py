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
import glob
import shutil
import json

import click
from torchvision import transforms as T

from .utils import set_logger
from .data_utils.aug import RandomStretchAug, RandomCrop, FgBgFlipAug
from .classification.dataset import ImageDataModule, read_tsv_file
from .classification.image_classifier import ImageClassifier, BASE_MODELS
from .trainer import PlTrainer, resave_model

_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}
logger = set_logger(log_level=logging.INFO)


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('train')
@click.option(
    '-c', '--categories', type=str, default='bad,good', help='分类包含的类别',
)
@click.option(
    '-b',
    '--base-model-name',
    type=click.Choice(BASE_MODELS.keys()),
    default='mobilenet_v2',
    help='使用的base模型名称',
)
@click.option(
    '-t', '--transform-configs', type=str, default=None, help='configs for transforms',
)
@click.option(
    '-i',
    '--index-dir',
    type=str,
    required=True,
    help='索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件',
)
@click.option(
    '--image-dir', type=str, default='.', required=True, help='图片所在的文件夹',
)
@click.option(
    '--train-config-fp',
    type=str,
    required=True,
    help='识别模型训练使用的json配置文件，参考 `docs/examples/train_config.json`',
)
@click.option(
    '-r',
    '--resume-from-checkpoint',
    type=str,
    default=None,
    help='恢复此前中断的训练状态，继续训练识别模型。所以文件中应该包含训练状态。默认为 `None`',
)
@click.option(
    '-p',
    '--pretrained-model-fp',
    type=str,
    default=None,
    help='导入的训练好的识别模型，作为模型初始值。'
    '优先级低于"--resume-from-checkpoint"，当传入"--resume-from-checkpoint"时，此传入失效。默认为 `None`',
)
def train(
    categories,
        base_model_name,
    transform_configs,
    index_dir,
    image_dir,
    train_config_fp,
    resume_from_checkpoint,
    pretrained_model_fp,
):
    """训练分类模型"""

    categories = [_c.strip() for _c in categories.split(',')]
    if transform_configs is not None:
        transform_configs = json.loads(transform_configs)
    model = ImageClassifier(
        base_model_name=base_model_name,
        categories=categories,
        transform_configs=transform_configs,
    )
    logger.info(model)
    train_config = json.load(open(train_config_fp))

    transform_configs = train_config.get('transform', dict())
    train_transform = T.Compose(
        [
            RandomStretchAug(min_ratio=0.2, max_ratio=1.2),
            RandomCrop((50, 50)),
            FgBgFlipAug(p=transform_configs.get('fgbgflip_prob', 0.0)),
            T.RandomInvert(p=0.2),
            T.RandomApply([T.RandomRotation(degrees=10)], p=0.4),
            T.RandomAutocontrast(p=0.05),
            # T.RandomPosterize(bits=4, p=0.3),
            # T.RandomAdjustSharpness(sharpness_factor=0.5, p=0.3),
            # T.RandomEqualize(p=0.3),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
            model.eval_transform,
        ]
    )
    val_transform = model.eval_transform

    data_mod = ImageDataModule(
        categories=categories,
        index_dir=index_dir,
        img_folder=image_dir,
        train_transforms=train_transform,
        val_transforms=val_transform,
        batch_size=train_config['batch_size'],
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory'],
    )

    trainer = PlTrainer(train_config, ckpt_fn=['image-clf'])
    # if pretrained_model_fp is not None:
    #     load_model_params(model, pretrained_model_fp)

    trainer.fit(
        model, datamodule=data_mod, resume_from_checkpoint=resume_from_checkpoint
    )


def write_pred_results(total_fps, results, out_thresholds, out_image_dir):
    for img_fp, res in zip(total_fps, results):
        pred, prob = res
        if prob < out_thresholds[0] or prob > out_thresholds[1]:
            continue
        dest_dir = os.path.join(out_image_dir, pred)
        os.makedirs(dest_dir, exist_ok=True)
        basename, suffix = os.path.basename(img_fp).rsplit('.', maxsplit=1)
        fn = f'{basename}-{pred}-{prob:.4f}.{suffix}'
        new_fp = os.path.join(dest_dir, fn)
        shutil.copy(img_fp, new_fp)


@cli.command('predict')
@click.option(
    '-c', '--categories', type=str, default='bad,good', help='分类包含的类别',
)
@click.option(
    '-b',
    '--base-model-name',
    type=click.Choice(BASE_MODELS.keys()),
    default='mobilenet_v2',
    help='使用的base模型名称',
)
@click.option(
    '-t', '--transform-configs', type=str, default=None, help='configs for transforms',
)
@click.option(
    '-m', '--model-fp', type=str, required=True, help='模型文件路径',
)
@click.option(
    "-d",
    "--device",
    help="使用cpu还是 `cuda` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为 `cpu`",
    type=str,
    default='cpu',
)
@click.option(
    '-i',
    '--index-fp',
    type=str,
    required=True,
    help='索引文件,如 train.tsv 和 dev.tsv 文件；或者文件所在的目录，此时预测此文件夹下的所有图片',
)
@click.option(
    '--image-dir',
    type=str,
    default=None,
    help='图片所在的文件夹(如果 `--index-fp` 传入的是文件夹，此参数无效)',
)
@click.option(
    '--batch-size', type=int, default=32, help='batch size for prediction',
)
@click.option(
    '--num-workers', type=int, default=0, help='num_workers for DataLoader',
)
@click.option(
    '--pin-memory', is_flag=True, help='pin_memory for DataLoader',
)
@click.option(
    '-o', '--out-image-dir', type=str, required=True, help='输出图片所在的文件夹',
)
@click.option(
    '--out-thresholds', type=str, default='0.0,1.0', help='仅输出预测概率值在此给定值范围内的图片',
)
def predict(
    categories,
        base_model_name,
        transform_configs,
    model_fp,
    device,
    index_fp,
    image_dir,
    batch_size,
    num_workers,
    pin_memory,
    out_image_dir,
    out_thresholds,
):
    """模型预测"""
    categories = [_c.strip() for _c in categories.split(',')]

    if transform_configs is not None:
        transform_configs = json.loads(transform_configs)

    model = ImageClassifier(
        base_model_name=base_model_name,
        categories=categories,
        transform_configs=transform_configs,
    )
    model.load(model_fp, device)

    if os.path.isfile(index_fp):
        img_fps, _ = read_tsv_file(index_fp, img_folder=image_dir, mode='eval')
        img_fps = [_fp for _fp in img_fps if _fp.endswith('g')]  # 仅预测图片，去掉视频和GIF
    elif os.path.isdir(index_fp):
        fn_list = glob.glob1(index_fp, '*g')
        img_fps = [os.path.join(index_fp, fn) for fn in fn_list]

    logger.info(f'totally, {len(img_fps)} images')

    img_preds = model.predict_images(
        img_fps, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory
    )

    os.makedirs(out_image_dir, exist_ok=True)
    out_thresholds = [float(v) for v in out_thresholds.split(',')]
    assert len(out_thresholds) == 2 and out_thresholds[0] <= out_thresholds[1]

    write_pred_results(img_fps, img_preds, out_thresholds, out_image_dir)


if __name__ == "__main__":
    cli()
