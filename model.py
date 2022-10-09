from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        models = {
            "rnn": nn.RNN(
                300,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            ),
            "gru": nn.GRU(
                300,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            ),
            "lstm": nn.LSTM(
                300,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            ),
        }
        self.recurrent = models["rnn"]
        if bidirectional:
            D = hidden_size * 2
        else:
            D = hidden_size
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(D),
            nn.LeakyReLU(),
            nn.Linear(D, D),
            nn.Dropout(dropout),
            nn.LayerNorm(D),
            nn.LeakyReLU(),
            nn.Linear(D, num_class),
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, x, **kwargs) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(x)
        x, h = self.recurrent(x, **kwargs)
        x = torch.sum(x, dim=1)
        x = self.mlp(x)
        return x


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
