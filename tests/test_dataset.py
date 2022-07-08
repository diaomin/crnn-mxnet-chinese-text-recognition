# coding: utf-8
import os
import sys
from pathlib import Path
import pytest

from torchvision import transforms

from cnocr.consts import VOCAB_FP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

EXAMPLE_DIR = Path(__file__).parent.parent / 'docs/examples'
INDEX_DIR = Path(__file__).parent.parent / 'data/test'

from cnocr.utils import save_img
from cnocr.data_utils.aug import NormalizeAug
from cnocr.dataset import OcrDataset, OcrDataModule

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


def test_ocr_dataset():
    train_ds = OcrDataset(INDEX_DIR / 'train.tsv', img_folder=EXAMPLE_DIR, mode='train')
    print(len(train_ds))
    print(train_ds[0])

    dev_ds = OcrDataset(INDEX_DIR / 'dev.tsv', img_folder=EXAMPLE_DIR, mode='dev')
    print(len(dev_ds))
    print(dev_ds[0])


def test_transformer():
    train_ds = OcrDataset(INDEX_DIR / 'train.tsv', img_folder=EXAMPLE_DIR, mode='train')
    if not os.path.exists('test-out'):
        os.makedirs('test-out')
    for i in range(min(20, len(train_ds))):
        img = train_ds[i][0]
        img = train_transform(img)
        save_img(img, f'test-out/{i}.png')


def test_ocr_data_module():
    data_mod = OcrDataModule(
        index_dir=INDEX_DIR,
        vocab_fp=VOCAB_FP,
        img_folder=EXAMPLE_DIR,
        train_transforms=train_transform,
        val_transforms=val_transform,
    )
    data_mod.setup()
    train_dataloader = data_mod.train_dataloader()
    batch = iter(train_dataloader).next()
    print(batch)
