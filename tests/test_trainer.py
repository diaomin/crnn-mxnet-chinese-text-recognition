# coding: utf-8
import os
import sys
from pathlib import Path
import pytest

from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

EXAMPLE_DIR = Path(__file__).parent.parent / 'examples'
INDEX_DIR = Path(__file__).parent.parent / 'data/test'
IMAGE_DIR = Path(__file__).parent.parent / 'data/images'

from cnocr.data_utils.aug import NormalizeAug
from cnocr.dataset import OcrDataModule
from cnocr.models.densenet import DenseNet
from cnocr.models.crnn import CRNN
from cnocr.trainer import PlTrainer

train_transform = transforms.Compose(
    [
        transforms.RandomInvert(p=0.5),
        transforms.RandomErasing(p=0.05),
        transforms.RandomRotation(degrees=2),
        transforms.RandomAutocontrast(p=0.05),
        NormalizeAug(),
    ]
)
val_transform = NormalizeAug()


def gen_model(vocab):
    net = DenseNet(32, [2, 2, 2, 2], 64)
    crnn = CRNN(net, vocab=vocab, lstm_features=512, rnn_units=128)
    return crnn


def test_trainer():
    data_mod = OcrDataModule(
        index_dir=INDEX_DIR,
        vocab_fp=EXAMPLE_DIR / 'label_cn.txt',
        img_folder=IMAGE_DIR,
        train_transforms=train_transform,
        val_transforms=val_transform,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    )
    # data_mod.setup()

    config = {
        'epochs': 2,
        'optimizer': 'adam',
        'learning_rate': 1e-5,
        "lr_scheduler": {
            "name": "multi_step",
            "step_size": 2,
            "gamma": 0.5
        },
        "precision": 32,
        "pl_checkpoint_monitor": "complete_match_epoch",
        "pl_checkpoint_mode": "max",
    }
    trainer = PlTrainer(config)
    model = gen_model(data_mod.vocab)
    trainer.fit(model, datamodule=data_mod)
