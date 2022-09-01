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
from pathlib import Path
from typing import Any, Optional, Union, List

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchmetrics

from .classification import ImageClassifier
from .lr_scheduler import get_lr_scheduler


logger = logging.getLogger(__name__)


def get_optimizer(name: str, model, learning_rate, weight_decay):
    r"""Init the Optimizer

    Returns:
        torch.optim: the optimizer
    """
    OPTIMIZERS = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'adagrad': optim.Adagrad,
        'rmsprop': optim.RMSprop,
    }

    try:
        opt_cls = OPTIMIZERS[name.lower()]
        optimizer = opt_cls(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    except:
        logger.warning('Received unrecognized optimizer, set default Adam optimizer')
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    return optimizer


class CompleteMatchMetric(object):
    def __init__(self, **kwargs):
        self.total_cnt = 0
        self.match_cnt = 0

    def __call__(self, preds, target):
        assert len(preds) == len(target)
        cur_match_cnt = sum([p == r for p, r in zip(preds, target)])
        cur_total_cnt = len(preds)
        self.total_cnt += cur_total_cnt
        self.match_cnt += cur_match_cnt
        return cur_match_cnt / (1e-6 + cur_total_cnt)

    def compute(self):
        return self.match_cnt / (1e-6 + self.total_cnt)


METRIC_MAPPING = {
    'accuracy': torchmetrics.Accuracy,
    'precision': torchmetrics.Precision,
    'recall': torchmetrics.Recall,
    'complete_match': CompleteMatchMetric,
}
try:
    METRIC_MAPPING['f1'] = torchmetrics.F1Score
    METRIC_MAPPING['cer'] = torchmetrics.CharErrorRate
except:
    pass


class Metrics(object):
    def __init__(self, configs=None):
        configs = configs or {'accuracy': {}}
        self._metrics = dict()
        for name, _config in configs.items():
            if name not in METRIC_MAPPING:
                logger.warning(f'metric {name} is not supported and will be ignored')
            self._metrics[name] = METRIC_MAPPING[name](**_config)
        if len(self._metrics) < 1:
            raise RuntimeError('no available metric is set, please check you `train_config.json`')

    @classmethod
    def from_config(cls, configs):
        return cls(configs)

    def compute(self):
        results = dict()
        for name, _metric in self._metrics.items():
            results[name] = float(_metric.compute())
        return results

    def add_batch(self, references, predictions):
        results = dict()
        for name, _metric in self._metrics.items():
            results[name] = float(_metric(preds=predictions, target=references))
        return results


class WrapperLightningModule(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model
        self._optimizer = get_optimizer(
            config['optimizer'],
            self.model,
            config['learning_rate'],
            config.get('weight_decay', 0),
        )

        self.train_metrics = self.get_metrics()
        self.val_metrics = self.get_metrics()

    def forward(self, x):
        return self.model(x)

    def get_metrics(self):
        return Metrics.from_config(self.config['metrics'])

    def _postprocess_preds(self, preds):
        if isinstance(self.model, ImageClassifier):
            return preds.detach().cpu()
        else:
            preds, _ = zip(*preds)
            return preds

    def training_step(self, batch, batch_idx):
        # if hasattr(self.model, 'set_current_epoch'):
        #     self.model.set_current_epoch(self.current_epoch)
        # else:
        #     setattr(self.model, 'current_epoch', self.current_epoch)
        res = self.model.calculate_loss(
            batch, return_model_output=True, return_preds=True
        )

        # update lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        losses = res['loss']
        preds = self._postprocess_preds(res['preds'])
        reals = res['target']
        if isinstance(reals, torch.Tensor):
            reals = reals.detach().cpu()
        train_metrics = self.train_metrics.add_batch(references=reals, predictions=preds)
        train_metrics['loss'] = losses.item()
        train_metrics = {
            f'train-{k}-step': v for k, v in train_metrics.items() if not np.isnan(v)
        }
        self.log_dict(
            train_metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True,
        )
        return losses

    def training_epoch_end(self, outputs) -> None:
        train_metrics = self.train_metrics.compute()
        train_metrics = {
            f'train-{k}-epoch': v for k, v in train_metrics.items() if not np.isnan(v)
        }
        train_metrics['train-loss-epoch'] = np.mean(
            [out['loss'].item() for out in outputs]
        )
        self.log_dict(
            train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )
        self.train_metrics = self.get_metrics()

    def validation_step(self, batch, batch_idx):
        # if hasattr(self.model, 'validation_step'):
        #     return self.model.validation_step(batch, batch_idx, self)

        res = self.model.calculate_loss(
            batch, return_model_output=True, return_preds=True
        )
        losses = res['loss']

        preds = self._postprocess_preds(res['preds'])
        reals = res['target']
        if isinstance(reals, torch.Tensor):
            reals = reals.detach().cpu()
        val_metrics = self.val_metrics.add_batch(references=reals, predictions=preds)
        val_metrics['loss'] = losses.item()
        # val_metrics['accuracy'] = sum([p == r for p, r in zip(preds, reals)]) / (len(reals) + 1e-6)

        # 过滤掉NaN的指标。有些指标在某些batch数据上会出现结果NaN，比如batch只有正样本或负样本时，AUC=NaN
        val_metrics = {
            f'val-{k}-step': v for k, v in val_metrics.items() if not np.isnan(v)
        }
        self.log_dict(
            val_metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True,
        )
        return losses

    def validation_epoch_end(self, outputs) -> None:
        val_metrics = self.val_metrics.compute()
        val_metrics = {
            f'val-{k}-epoch': v for k, v in val_metrics.items() if not np.isnan(v)
        }
        val_metrics['val-loss-epoch'] = np.mean([out.item() for out in outputs])
        self.log_dict(
            val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )
        self.val_metrics = self.get_metrics()

    def configure_optimizers(self):
        return [self._optimizer], [get_lr_scheduler(self.config, self._optimizer)]


class PlTrainer(object):
    """
    封装 PyTorch Lightning 的训练器。
    """

    def __init__(self, config, ckpt_fn=None):
        self.config = config

        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [lr_monitor]

        mode = self.config.get('pl_checkpoint_mode', 'min')
        monitor = self.config.get('pl_checkpoint_monitor')
        fn_fields = ckpt_fn or []
        fn_fields.append('{epoch:03d}')
        if monitor:
            fn_fields.append('{' + monitor + ':.4f}')
            checkpoint_callback = ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                filename='-'.join(fn_fields),
                save_last=True,
                save_top_k=5,
            )
            callbacks.append(checkpoint_callback)

        self.pl_trainer = pl.Trainer(
            limit_train_batches=self.config.get('limit_train_batches', 1.0),
            limit_val_batches=self.config.get('limit_val_batches', 1.0),
            accelerator=self.config.get('accelerator', 'auto'),
            devices=self.config.get('devices'),
            max_epochs=self.config.get('epochs', 20),
            precision=self.config.get('precision', 32),
            log_every_n_steps=self.config.get('log_every_n_steps', 10),
            callbacks=callbacks,
        )

    def fit(
        self,
        model: nn.Module,
        train_dataloader: Any = None,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
    ):
        r"""
        Runs the full optimization routine.

        Args:
            model: Model to fit.

            train_dataloader: Either a single PyTorch DataLoader or a collection of these
                (list, dict, nested lists and dicts). In the case of multiple dataloaders, please
                see this :ref:`page <multiple-training-dataloaders>`
            val_dataloaders: Either a single Pytorch Dataloader or a list of them, specifying validation samples.
                If the model has a predefined val_dataloaders method this will be skipped
            datamodule: A instance of :class:`LightningDataModule`.
            resume_from_checkpoint: Path/URL of the checkpoint from which training is resumed. If there is
                no checkpoint file at the path, start from scratch. If resuming from mid-epoch checkpoint,
                training will start from the beginning of the next epoch.
        """
        steps_per_epoch = (
            len(train_dataloader)
            if train_dataloader is not None
            else len(datamodule.train_dataloader())
        )
        steps_per_epoch += (
            len(val_dataloaders)
            if val_dataloaders is not None
            else len(datamodule.val_dataloader())
        )
        self.config['steps_per_epoch'] = steps_per_epoch
        if resume_from_checkpoint is not None:
            pl_module = WrapperLightningModule.load_from_checkpoint(
                resume_from_checkpoint, config=self.config, model=model
            )
            self.pl_trainer = pl.Trainer(resume_from_checkpoint=resume_from_checkpoint)
        else:
            pl_module = WrapperLightningModule(self.config, model)

        self.pl_trainer.fit(pl_module, train_dataloader, val_dataloaders, datamodule)

        fields = self.pl_trainer.checkpoint_callback.best_model_path.rsplit(
            '.', maxsplit=1
        )
        fields[0] += '-model'
        output_model_fp = '.'.join(fields)
        resave_model(
            self.pl_trainer.checkpoint_callback.best_model_path, output_model_fp
        )
        self.saved_model_file = output_model_fp


def resave_model(module_fp, output_model_fp, map_location=None):
    """PlTrainer存储的文件对应其 `pl_module` 模块，需利用此函数转存为 `model` 对应的模型文件。"""
    checkpoint = torch.load(module_fp, map_location=map_location)
    state_dict = {}
    if all([k.startswith('model.') for k in checkpoint['state_dict'].keys()]):
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.split('.', maxsplit=1)[1]] = v
    torch.save({'state_dict': state_dict}, output_model_fp)
