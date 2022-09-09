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

import logging
from typing import Tuple, List, Union, Optional

import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v2, densenet121
from torchvision.transforms import functional as F, InterpolationMode

from ..utils import read_img, load_model_params

logger = logging.getLogger(__name__)


BASE_MODELS = {
    'mobilenet_v2': [mobilenet_v2, None],
    'densenet121': [densenet121, None],
}

try:
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    BASE_MODELS['efficientnet_v2_s'] = [
        efficientnet_v2_s,
        EfficientNet_V2_S_Weights.DEFAULT,
    ]
except:
    pass

try:
    from torchvision.models import (
        MobileNet_V2_Weights,
        DenseNet121_Weights,
    )

    BASE_MODELS['mobilenet_v2'][1] = MobileNet_V2_Weights.DEFAULT
    BASE_MODELS['densenet121'][1] = DenseNet121_Weights.DEFAULT
except:
    pass


class ImageTransform(nn.Module):
    # copy from MobileNet_V2_Weights.weights.transforms
    def __init__(
        self,
        *,
        resize_size: int = 232,
        crop_size: Union[int, List[int]] = 224,
        resize_max_size: Optional[int] = None,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size] if isinstance(crop_size, int) else crop_size
        self.resize_size = [resize_size]
        self.resize_max_size = resize_max_size
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): a tensor with values ranging from 0 to 255,
                and with shape [3, height, width].

        Returns: a normalized tensor with shape [3, height, width]

        """
        img = F.resize(
            img,
            self.resize_size,
            max_size=self.resize_max_size,
            interpolation=self.interpolation,
        )
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, torch.Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )


class ImageClassifier(nn.Module):
    def __init__(self, base_model_name: str, categories, *, transform_configs=None):
        super().__init__()
        transform_configs = transform_configs or dict()
        self.eval_transform = ImageTransform(**transform_configs)

        self.categories = categories
        assert len(self.categories) > 1

        try:
            model_cls, model_weights = BASE_MODELS[base_model_name]
        except:
            logger.warning(
                f'model {base_model_name} is not supported yet. Use default model `mobilenet_v2` instead'
            )
            model_cls, model_weights = BASE_MODELS['mobilenet_v2']

        if model_weights is not None:
            self.base = model_cls(weights=model_weights)
        else:
            self.base = model_cls(pretrained=True)

        if 'densenet' in base_model_name:
            last_channel = self.base.classifier.in_features
            dropout = 0.0
        else:
            dropout = self.base.classifier[0].p
            last_channel = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(last_channel, len(self.categories)),
        )

        self.criterion = nn.CrossEntropyLoss()

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def load(self, model_fp, device='cpu'):
        self.device = torch.device(device)
        self.to(self.device)
        load_model_params(self, model_fp, device=device)

    def forward(self, x: torch.Tensor):
        logits = self.base(x)
        prediction = logits.softmax(dim=1)
        pred_probs, preds = prediction.topk(1, dim=1)
        outs = dict(
            logits=logits, preds=preds.squeeze(-1), probs=pred_probs.squeeze(-1)
        )
        return outs

    def calculate_loss(self, batch, **kwargs):
        imgs, labels = batch
        outs = self(imgs)
        loss = self.criterion(outs['logits'], labels)

        outs.update(dict(target=labels, loss=loss))
        return outs

    @torch.no_grad()
    def predict_images(
        self,
        images: List[Union[str, torch.Tensor]],
        *,
        batch_size: int = 32,
        **loader_kwargs,
    ) -> List[Tuple[str, float]]:
        """
        预测给定图片列表的类别。

        Args:
            images (List[Union[str, torch.Tensor]]): if is a torch.Tensor, the tensor should be
                with values ranging from 0 to 255, and with shape [height, width, 3].
            batch_size (int): batch size. Default: 32.
            **loader_kwargs ():

        Returns: [(<类别名称>, <对应概率>), (<类别名称>, <对应概率>), (<类别名称>, <对应概率>), ...]

        """
        self.eval()

        def collate_fn(_images):
            img_list = [
                self.eval_transform(_img) for _img in _images if _img is not None
            ]
            imgs = torch.stack(img_list)
            goods = torch.tensor(
                [idx for idx, _img in enumerate(_images) if _img is not None],
                dtype=torch.int32,
            )
            return imgs, goods, torch.tensor(len(_images))

        class ListDataset(Dataset):
            def __init__(self, _images):
                self._images = _images

            def __getitem__(self, idx):
                img = self._images[idx]
                if isinstance(img, str):
                    try:
                        img = torch.tensor(
                            read_img(img, gray=False).transpose((2, 0, 1))
                        )
                    except:
                        img = None
                elif isinstance(img, torch.Tensor):
                    img = img.permute((2, 0, 1))
                return img

            def __len__(self):
                return len(self._images)

        dataset = ListDataset(images)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs,
        )

        results = []
        for batch in tqdm.tqdm(dataloader):
            _images = batch[0].to(self.device)
            outs = self(_images)
            goods = batch[1].cpu().numpy().tolist()
            batch_len = int(batch[2].cpu())
            part_res = [(self.categories[0], 1.0 / len(self.categories))] * batch_len
            for idx, pred, prob in zip(
                goods,
                outs['preds'].cpu().numpy().tolist(),
                outs['probs'].cpu().numpy().tolist(),
            ):
                part_res[idx] = (self.categories[pred], prob)
            results.extend(part_res)
        return results


if __name__ == '__main__':
    img = torch.tensor(
        read_img("dev-samples/dev-0.png", gray=False).transpose((2, 0, 1))
    )

    transform_configs = {
        'crop_size': [150, 450],
        'resize_size': 160,
        'resize_max_size': 1000,
    }
    clf = ImageClassifier(
        base_model_name='mobilenet_v2',
        categories=('bad', 'good'),
        transform_configs=transform_configs,
    )
    clf.eval()
    preprocess = clf.eval_transform
    batch = preprocess(img).unsqueeze(0)

    outs = clf(batch)

    class_id = outs['preds'][0].item()
    score = outs['probs'][0].item()
    category_name = clf.categories[class_id]
    print(f"{category_name}: {100 * score:.1f}%")
