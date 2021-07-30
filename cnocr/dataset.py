# coding: utf-8
from pathlib import Path
from typing import Optional, Union, List, Tuple, Callable

import pytorch_lightning as pt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import resize

from .consts import IMG_STANDARD_HEIGHT
from .utils import read_charset, read_tsv_file, read_img


class OcrDataset(Dataset):
    def __init__(self, index_fp, img_folder=None, mode='train'):
        super().__init__()
        # self.vocab = vocab
        # self.letter2id = {letter: idx for idx, letter in enumerate(self.vocab)}
        self.img_fp_list, self.labels_list = read_tsv_file(
            index_fp, '\t', img_folder, mode
        )
        self.mode = mode

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, item):
        img_fp = self.img_fp_list[item]
        img = read_img(img_fp)
        ori_height, ori_width = img.shape[1:]
        ratio = ori_height / IMG_STANDARD_HEIGHT
        img = torch.from_numpy(img)
        if img.size(1) != IMG_STANDARD_HEIGHT:
            img = resize(img, [IMG_STANDARD_HEIGHT, int(ori_width / ratio)])

        if self.mode != 'test':
            labels = self.labels_list[item]
            # label_ids = [self.letter2id[l] for l in labels]

        return (img, labels) if self.mode != 'test' else (img,)


def _pad_seq(img_list):
    """
    Pad a list of variable width image Tensors with ``padding_value`.

    :param img_list: [C, H, W], where W is variable width
    :return: [B, C, H, W_max]
    """
    img_list = [img.permute((2, 0, 1)) for img in img_list]  # [W, C, H]
    imgs = pad_sequence(img_list, batch_first=True, padding_value=0)  # [B, W_max, C, H]
    return imgs.permute((0, 2, 3, 1))  # [B, C, H, W_max]


def collate_fn(img_labels: List[Tuple[str, str]], transformers: Callable = None):
    test_mode = len(img_labels[0]) == 1
    if test_mode:
        img_list = zip(*img_labels)
        labels_list, label_lengths = None, None
    else:
        img_list, labels_list = zip(*img_labels)
        label_lengths = torch.tensor([len(labels) for labels in labels_list])

    img_lengths = torch.tensor([img.size(2) for img in img_list])
    if transformers is not None:
        img_list = [transformers(img) for img in img_list]
    imgs = _pad_seq(img_list)
    return imgs, img_lengths, labels_list, label_lengths


class OcrDataModule(pt.LightningDataModule):
    def __init__(
        self,
        index_dir: Union[str, Path],
        vocab_fp: Union[str, Path],
        img_folder: Union[str, Path, None] = None,
        train_transforms=None,
        val_transforms=None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(
            train_transforms=train_transforms, val_transforms=val_transforms
        )
        self.vocab, self.letter2id = read_charset(vocab_fp)
        self.index_dir = Path(index_dir)
        self.img_folder = img_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train = OcrDataset(
            self.index_dir / 'train.tsv', self.img_folder, mode='train'
        )
        self.val = OcrDataset(self.index_dir / 'dev.tsv', self.img_folder, mode='train')

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
        cur_collate_fn = lambda x: collate_fn(x, self.train_transforms)
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=cur_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        cur_collate_fn = lambda x: collate_fn(x, self.val_transforms)
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=cur_collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return None
