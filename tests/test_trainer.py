# coding: utf-8
import os
import sys
from pathlib import Path
import pytest

from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

EXAMPLE_DIR = Path(__file__).parent.parent / 'docs/examples'
INDEX_DIR = Path(__file__).parent.parent / 'data/test'
IMAGE_DIR = Path(__file__).parent.parent / 'data/images'

from cnocr import gen_model
from cnocr.consts import VOCAB_FP
from cnocr.data_utils.aug import NormalizeAug
from cnocr.dataset import OcrDataModule
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


def test_trainer():
    data_mod = OcrDataModule(
        index_dir=INDEX_DIR,
        vocab_fp=VOCAB_FP,
        img_folder=IMAGE_DIR,
        train_transforms=train_transform,
        val_transforms=val_transform,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    )

    config = {
        'epochs': 2,
        'optimizer': 'adam',
        'learning_rate': 1e-5,
        "lr_scheduler": {
            "name": "cos_warmup",
            "min_lr_mult_factor": 0.01,
            "warmup_epochs": 0.2
        },
        "precision": 32,
        "pl_checkpoint_monitor": "complete_match_epoch",
        "pl_checkpoint_mode": "max",
    }
    trainer = PlTrainer(config)
    model = gen_model('densenet_lite_136-fc', data_mod.vocab)
    trainer.fit(model, datamodule=data_mod)
