# coding: utf-8
from itertools import groupby
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..data_utils.utils import encode_sequences
from ..utils import gen_length_mask

__all__ = ['CRNN', 'CTCPostProcessor']


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
        raise NotImplementedError()

    @property
    def compress_ratio(self):
        raise NotImplementedError()


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

    @property
    def compress_ratio(self):
        return self.feat_extractor.compress_ratio

    def calculate_loss(
        self, batch, return_model_output: bool = False, return_preds: bool = False,
    ):
        imgs, img_lengths, labels_list, label_lengths = batch
        return self(
            imgs,
            img_lengths,
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
        input_lengths: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        features = self.feat_extractor(x)
        input_lengths = input_lengths // self.compress_ratio
        # B x C x H x W --> B x C*H x W --> B x W x C*H
        c, h, w = features.shape[1], features.shape[2], features.shape[3]
        features_seq = torch.reshape(features, shape=(-1, h * c, w))
        features_seq = torch.transpose(features_seq, 1, 2)
        features_seq = pack_padded_sequence(features_seq, input_lengths, batch_first=True, enforce_sorted=False)
        logits, _ = self.decoder(features_seq)
        logits, output_lens = pad_packed_sequence(logits, batch_first=True, total_length=w)
        logits = self.linear(logits)

        out: Dict[str, Any] = {}
        if return_model_output:
            out["logits"] = logits

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(logits, input_lengths)

        if target is not None:
            out['loss'] = self._compute_loss(logits, target, input_lengths)

        return out
