import math

import torch
import torch.nn as nn

from .positional_encoder import PositionalEncoder
from .state_encoder import StateEncoder


class ExplainerTransformer(nn.Module):
    def __init__(
            self,
            d_state: int,
            tgt_vocab_size:int,
            d_hidden: int=512,
            nhead: int=8,
            num_decoder_layers:int=6,
            dim_feedforward: int=2048,
            max_len=500,
            dropout: float=0.1,
            batch_first: bool=False,
            bias: bool=True
    ):
        super().__init__()
        self.d_hidden = d_hidden

        self.embedding = nn.Embedding(tgt_vocab_size, d_hidden)

        self.pe = PositionalEncoder(d_state, dropout, max_len)

        self.encoder = StateEncoder(
            d_state,
            d_hidden,
            nhead,
            self.pe,
            dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias
        )

        self.transformer = nn.Transformer(
            d_hidden,
            nhead,
            num_decoder_layers=num_decoder_layers,
            num_encoder_layers=num_decoder_layers-1,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias
        )

        self.output_layer = nn.Linear(d_hidden, tgt_vocab_size)
    
    def forward(self, tail: torch.Tensor, seq: torch.Tensor, tgt: torch.Tensor):
        """
        Logits of the next token on the output.
        """
        src_embedded = self.encoder(tail, seq)
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_hidden)
        transformer_out = self.transformer(src_embedded, tgt_embedded)

        result = self.output_layer(transformer_out)
        return result