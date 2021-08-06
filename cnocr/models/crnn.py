# coding: utf-8
from itertools import groupby

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Dict, Any, Optional, List

__all__ = ['CRNN', 'CTCPostProcessor']

# default_cfgs: Dict[str, Dict[str, Any]] = {
#     'crnn_vgg16_bn': {
#         'mean': (.5, .5, .5),
#         'std': (1., 1., 1.),
#         'backbone': 'vgg16_bn', 'rnn_units': 128, 'lstm_features': 512,
#         'input_shape': (3, 32, 128),
#         'vocab': VOCABS['french'],
#         'url': None,
#     },
#     'crnn_resnet31': {
#         'mean': (.5, .5, .5),
#         'std': (1., 1., 1.),
#         'backbone': 'resnet31', 'rnn_units': 128, 'lstm_features': 4 * 512,
#         'input_shape': (3, 32, 128),
#         'vocab': VOCABS['french'],
#         'url': None,
#     },
# }
from ..data_utils.utils import encode_sequences
from ..utils import gen_length_mask


class CTCPostProcessor(object):
    """
    Postprocess raw prediction of the model (logits) to a list of words using CTC decoding

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(self, vocab: List[str],) -> None:

        self.vocab = vocab
        self._embedding = list(self.vocab) + ['<eos>']

    @staticmethod
    def ctc_best_path(
        logits: torch.Tensor,
        vocab: List[str],
        input_lengths: Optional[torch.Tensor] = None,
        blank: int = 0,
    ) -> List[Tuple[List[str], float]]:
        """Implements best path decoding as shown by Graves (Dissertation, p63), highly inspired from
        <https://github.com/githubharald/CTCDecoder>`_.

        Args:
            logits: model output, shape: N x T x C
            vocab: vocabulary to use
            input_lengths: valid sequence lengths
            blank: index of blank label

        Returns:
            A list of tuples: (word, confidence)
        """
        # compute softmax
        probs = F.softmax(logits.permute(0, 2, 1), dim=1)
        # get char indices along best path
        best_path = torch.argmax(probs, dim=1)  # [N, T]

        if input_lengths is not None:
            length_mask = gen_length_mask(input_lengths, probs.shape)  # [N, 1, T]
            probs.masked_fill_(length_mask, 1.0)
            best_path.masked_fill_(length_mask.squeeze(1), blank)

        # define word proba as min proba of sequence
        probs, _ = torch.max(probs, dim=1)  # [N, T]
        probs, _ = torch.min(probs, dim=1)  # [N]

        words = []
        for sequence in best_path:
            # collapse best path (using itertools.groupby), map to chars, join char list to string
            collapsed = [vocab[k] for k, _ in groupby(sequence) if k != blank]
            words.append(collapsed)

        return list(zip(words, probs.tolist()))

    def __call__(  # type: ignore[override]
        self, logits: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> List[Tuple[List[str], float]]:
        """
        Performs decoding of raw output with CTC and decoding of CTC predictions
        with label_to_idx mapping dictionnary

        Args:
            logits: raw output of the model, shape (N, C + 1, seq_len)
            input_lengths: valid sequence lengths

        Returns:
            A tuple of 2 lists: a list of str (words) and a list of float (probs)

        """
        # Decode CTC
        return self.ctc_best_path(
            logits=logits,
            vocab=self.vocab,
            input_lengths=input_lengths,
            blank=len(self.vocab),
        )


class OcrModel(nn.Module):
    def calculate_loss(self, batch):
        ...


class CRNN(OcrModel):
    """Implements a CRNN architecture as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of units in the LSTM layers
        cfg: configuration dictionary
    """

    _children_names: List[str] = [
        'feat_extractor',
        'decoder',
        'linear',
        'postprocessor',
    ]

    def __init__(
        self,
        feature_extractor: nn.Module,
        vocab: List[str],
        lstm_features: int,
        rnn_units: int = 128,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.letter2id = {letter: idx for idx, letter in enumerate(self.vocab)}
        assert len(self.vocab) == len(self.letter2id)
        self.cfg = cfg
        self.feat_extractor = feature_extractor

        self.decoder = nn.LSTM(
            input_size=lstm_features,
            hidden_size=rnn_units,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
        )

        # features units = 2 * rnn_units because bidirectional layers
        self.linear = nn.Linear(in_features=2 * rnn_units, out_features=len(vocab) + 1)

        self.postprocessor = CTCPostProcessor(vocab=vocab)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def calculate_loss(
        self, batch, return_model_output: bool = False, return_preds: bool = False,
    ):
        imgs, img_lengths, labels_list, label_lengths = batch
        COMPRESS_VAL = 8
        return self(
            imgs,
            img_lengths // COMPRESS_VAL,
            labels_list,
            return_model_output,
            return_preds,
        )

    def _compute_loss(
        self,
        model_output: torch.Tensor,
        target: List[str],
        seq_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute CTC loss for the model.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_length: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        gt, seq_len = self.compute_target(target)

        if seq_length is None:
            batch_len = model_output.shape[0]
            seq_length = model_output.shape[1] * torch.ones(
                size=(batch_len,), dtype=torch.int32
            )

        # N x T x C -> T x N x C
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)

        ctc_loss = F.ctc_loss(
            probs,
            torch.from_numpy(gt).to(device=probs.device),
            seq_length,
            torch.tensor(seq_len, dtype=torch.int, device=probs.device),
            len(self.vocab),
        )

        return ctc_loss

    def compute_target(self, gts: List[str],) -> Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
            gts: list of ground-truth labels

        Returns:
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(
            sequences=gts, vocab=self.letter2id, eos=len(self.letter2id),
        )
        seq_len = [len(word) for word in gts]
        return encoded, seq_len

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor = None,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        features = self.feat_extractor(x)
        # B x C x H x W --> B x C*H x W --> B x W x C*H
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)
        logits, _ = self.decoder(features_seq)
        logits = self.linear(logits)

        out: Dict[str, Any] = {}
        if return_model_output:
            out["out_map"] = logits

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(logits, input_lengths)

        if target is not None:
            out['loss'] = self._compute_loss(logits, target, input_lengths)

        return out


# def _crnn(arch: str, pretrained: bool, input_shape: Optional[Tuple[int, int, int]] = None, **kwargs: Any) -> CRNN:
#
#     # Patch the config
#     _cfg = deepcopy(default_cfgs[arch])
#     _cfg['input_shape'] = input_shape or _cfg['input_shape']
#     _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])
#     _cfg['rnn_units'] = kwargs.get('rnn_units', _cfg['rnn_units'])
#
#     # Feature extractor
#     feat_extractor = backbones.__dict__[_cfg['backbone']]()
#
#     kwargs['vocab'] = _cfg['vocab']
#     kwargs['rnn_units'] = _cfg['rnn_units']
#     kwargs['lstm_features'] = _cfg['lstm_features']
#
#     # Build the model
#     model = CRNN(feat_extractor, cfg=_cfg, **kwargs)
#     # Load pretrained parameters
#     if pretrained:
#         raise NotImplementedError
#
#     return model
#
#
# def crnn_vgg16_bn(pretrained: bool = False, **kwargs: Any) -> CRNN:
#     """CRNN with a VGG-16 backbone as described in `"An End-to-End Trainable Neural Network for Image-based
#     Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.
#
#     Example::
#         >>> import torch
#         >>> from doctr.models import crnn_vgg16_bn
#         >>> model = crnn_vgg16_bn(pretrained=True)
#         >>> input_tensor = torch.rand(1, 3, 32, 128)
#         >>> out = model(input_tensor)
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
#
#     Returns:
#         text recognition architecture
#     """
#
#     return _crnn('crnn_vgg16_bn', pretrained, **kwargs)
#
#
# def crnn_resnet31(pretrained: bool = False, **kwargs: Any) -> CRNN:
#     """CRNN with a ResNet-31 backbone as described in `"An End-to-End Trainable Neural Network for Image-based
#     Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.
#
#     Example::
#         >>> import torch
#         >>> from doctr.models import crnn_resnet31
#         >>> model = crnn_resnet31(pretrained=True)
#         >>> input_tensor = torch.rand(1, 3, 32, 128)
#         >>> out = model(input_tensor)
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
#
#     Returns:
#         text recognition architecture
#     """
#
#     return _crnn('crnn_resnet31', pretrained, **kwargs)
