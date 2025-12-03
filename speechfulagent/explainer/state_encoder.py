import math

import torch
import torch.nn as nn

from .positional_encoder import PositionalEncoder


class StateEncoder(nn.Module):
    def __init__(
            self, 
            d_state: int,
            d_hidden: int,
            nhead: int, 
            pos_encoder: PositionalEncoder,
            dim_feedforward: int=2048,
            dropout: float=0.1,
            batch_first: bool=False,
            bias: bool=True
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.pe = pos_encoder

        self.activation = nn.ReLU()

        self.resize_tail = nn.Linear(d_state, d_hidden, bias=bias)
        self.resize_sequence = nn.Linear(d_state, d_hidden, bias=bias)

        self.attn = TailMultiheadAttention(
            d_hidden,
            nhead,
            batch_first=batch_first,
            bias=bias
        )

        self.feed_forward = FeedForward(
            d_hidden,
            dim_feedforward,
            dropout=dropout,
            bias=bias
        )
        self.norm = nn.LayerNorm(d_hidden, bias=bias)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tail: torch.Tensor, sequence: torch.Tensor):
        tail = self.activation(self.resize_tail(tail)) * math.sqrt(self.d_hidden)
        tail = self.pe(tail)
        sequence = self.activation(self.resize_sequence(sequence)) * math.sqrt(self.d_hidden)
        sequence = self.pe(sequence)

        attn_output = self.attn(tail, sequence)

        feed_forward = self.feed_forward(attn_output)
        result = self.norm(attn_output + self.dropout(feed_forward))

        return result

class TailMultiheadAttention(nn.Module):
    def __init__(
            self,
            d_hidden: int,
            nhead: int,
            batch_first: bool=False,
            bias: bool=True
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_hidden, d_hidden, bias=bias)

        self.linear2 = nn.Linear(d_hidden, d_hidden, bias=bias)

        self.attn1 = nn.MultiheadAttention(
            d_hidden,
            nhead,
            batch_first=batch_first,
            bias=bias
        )
        self.norm1 = nn.LayerNorm(d_hidden, bias=bias)

        self.attn2 = nn.MultiheadAttention(
            d_hidden,
            nhead,
            batch_first=batch_first,
            bias=bias
        )
        self.norm2 = nn.LayerNorm(d_hidden, bias=bias)

        self.linear_out = nn.Linear(d_hidden, d_hidden, bias=bias)

    def forward(self, tail: torch.Tensor, sequence: torch.Tensor):
        tail_lin = self.linear1(tail)
        tail_attn, _ = self.attn1(tail_lin, tail_lin, tail_lin)
        tail = self.norm1(tail + tail_attn)

        sequence_lin = self.linear2(sequence)
        sequence_attn, _ = self.attn2(sequence_lin, sequence_lin, sequence_lin)
        sequence = self.norm2(sequence + sequence_attn)

        attn_sum = tail + sequence
        return self.linear_out(attn_sum)


class FeedForward(nn.Module):
    def __init__(
            self,
            d_hidden: int,
            d_feedforward: int,
            dropout: float=0.1,
            bias: bool=True
    ):
        super().__init__()
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_hidden, d_feedforward, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_feedforward, d_hidden, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))