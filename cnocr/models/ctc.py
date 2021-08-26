# coding: utf-8
# Copyright (C) 2021, [Breezedeus](https://github.com/breezedeus).
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
# Credits: adapted from https://github.com/mindee/doctr

from itertools import groupby
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F

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
            length_mask = gen_length_mask(input_lengths, probs.shape).to(
                device=probs.device
            )  # [N, 1, T]
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
