import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import math
import random
import pandas as pd
import datetime
import time
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
# feed forward

class FeedForward(nn.Module):
    def __init__(self, d_model, d_middle, dropout):
        super().__init__()
        self.l_feed = nn.Linear(d_model, d_middle)
        self.l_proj = nn.Linear(d_middle, d_model)
        self.dropout = nn.Dropout(dropout)
        self.NG = NewGELU()

    def forward(self, x):
        return self.dropout(self.l_proj(self.NG(self.l_feed(x))))

class AttentionModule(nn.Module):
    def __init__(self, d_model, d_Q, d_K, d_V, dropout):
        super().__init__()
        self.Q = nn.Linear(d_model, d_Q, bias=False)
        self.K = nn.Linear(d_model, d_K, bias=False)
        self.V = nn.Linear(d_model, d_V, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        y = self.attention(self.Q(q), self.K(k), self.V(v), mask)
        return y

    def attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
        if mask is not None:
            scores += torch.triu(torch.ones_like(scores) * float("-inf"), diagonal=1)  # mask subsequent positions
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return attn @ V

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, d_model, h, d_Q, d_K, d_V, dropout):
        super().__init__()
        self.l_proj = nn.Linear(h * d_V, d_model)
        self.a_modules = nn.ModuleList(AttentionModule(d_model, d_Q, d_K, d_V, dropout) for _ in range(h))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # collect all the heads together and project. Each head has shape (b, L, d_v)
        # the concatenated shape is (b, L, h * d_v)
        # This is projected to (b, L, d_v) This is the end result of the MHAM.
        return self.dropout(self.l_proj(torch.cat([layer(q, k, v) for layer in self.a_modules], dim=-1)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_middle, dropout, h, d_Q, d_K, d_V):
        super().__init__()

        # multihead
        self.multi_head = MultiHeadAttentionModule(d_model, h, d_Q, d_K, d_V)
        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # feed forward
        self.feed_forward = FeedForward(d_model, d_middle)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # multi head and skip and add
        x = x + self.dropout(self.multi_head(x, x, x, mask))
        # take the norm
        x = self.norm1(x)
        # feed forward and skip and add
        x = x + self.dropout(self.feed_forward(x))
        # take the norm
        x = self.norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_blocks, d_model, d_middle, dropout, h, d_Q, d_K, d_V):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderBlock(d_model, d_middle, dropout, h, d_Q, d_K, d_V) for _ in range(num_blocks))

    def forward(self, x, pe, mask=None):
        x += pe
        for layer in self.layers:
            x = layer(x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_middle, dropout, h, d_Q, d_K, d_V):
        super().__init__()

        # multihead_masked
        self.multi_head_masked = MultiHeadAttentionModule(d_model, h, d_Q, d_K, d_V)
        # multihead_encoder
        self.multi_head_encoder = MultiHeadAttentionModule(d_model, h, d_Q, d_K, d_V)
        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # feed forward
        self.feed_forward = FeedForward(d_model, d_middle)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask):
        """
        x: decoder input
        y: encoder output
        """
        x = x + self.dropout(self.multi_head_masked(x, x, x, mask))
        x = self.norm1(x)
        x = x + self.dropout(self.multi_head_encoder(x, y, y, None))
        x = self.norm2(x)
        x = x + self.dropout(self.feed_forward(x))
        x = self.norm3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_blocks, d_model, d_middle, d_token, dropout, h, d_Q, d_K, d_V):
        super().__init__()

        self.layers = nn.ModuleList(
            DecoderBlock(d_model, d_middle, dropout, h, d_Q, d_K, d_V) for _ in range(num_blocks))

        self.l1 = nn.Linear(d_model, d_token)

    def forward(self, enc_out, dec_inp, pe, dec_mask=None):
        x = dec_inp + pe
        y = enc_out
        for layer in self.layers:
            x = layer(x, y, dec_mask)
        x = self.l1(x)

        return x

class Transformer(nn.Module):
    def __init__(self, num_blocks, d_model, d_middle, d_token, dropout, h, d_Q, d_K, d_V):
        super().__init__()
        self.encoder = Encoder(num_blocks, d_model, d_middle, dropout, h, d_Q, d_K, d_V)
        self.decoder = Decoder(num_blocks, d_model, d_middle, d_token, dropout, h, d_Q, d_K, d_V)

    def forward(self, enc_src, dec_src, pe, enc_mask, dec_mask):
        return self.decoder(self.encoder(enc_src, pe, enc_mask), dec_src, pe, dec_mask)






