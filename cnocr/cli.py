# coding: utf-8
from __future__ import absolute_import, division, print_function
import logging
import click
import json

import torch
from torchvision import transforms

from cnocr.utils import set_logger
from cnocr.data_utils.aug import NormalizeAug
from cnocr.dataset import OcrDataModule
from cnocr.models.densenet import DenseNet
from cnocr.models.crnn import CRNN
from cnocr.trainer import PlTrainer

_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}
logger = set_logger(log_level=logging.INFO)


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('train')
@click.option(
    '--index-dir',
    type=str,
    required=True,
    help='索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件',
)
@click.option('--train-config-fp', type=str, required=True, help='训练使用的json配置文件')
@click.option(
    '-m', '--pretrained-model-fp', type=str, default=None, help='导入的训练好的模型，作为初始模型'
)
def train(index_dir, train_config_fp, pretrained_model_fp):
    train_transform = transforms.Compose(
        [
            transforms.RandomInvert(p=0.5),
            # transforms.RandomErasing(p=0.05, scale=(0.01, 0.05)),
            transforms.RandomRotation(degrees=2),
            transforms.RandomAutocontrast(p=0.05),
            NormalizeAug(),
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

    trainer = PlTrainer(train_config)
    model = gen_model(data_mod.vocab)

    if pretrained_model_fp is not None:
        checkpoint = torch.load(pretrained_model_fp, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        if all([param_name.startswith('model.') for param_name in state_dict.keys()]):
            # 表示导入的模型是通过 PlTrainer 训练出的 WrapperLightningModule，对其进行转化
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k.split('.', maxsplit=1)[1]] = v
        model.load_state_dict(state_dict)

    trainer.fit(model, datamodule=data_mod)


def gen_model(vocab):
    net = DenseNet(32, [2, 2, 2, 2], 64)
    crnn = CRNN(net, vocab=vocab, lstm_features=512, rnn_units=128)
    return crnn


if __name__ == "__main__":
    cli()
