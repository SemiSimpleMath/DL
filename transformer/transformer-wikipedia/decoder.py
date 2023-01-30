import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# feed forward

class FeedForward(nn.Module):
    def __init__(self, d_model, d_middle, dropout):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_middle)
        self.l2 = nn.Linear(d_middle, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.dropout(F.relu(self.l1(x))))  # added drop out


def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
    scores += torch.triu(torch.ones_like(scores) * float("-inf"), diagonal=1)  # mask subsequent positions
    attn = F.softmax(scores, dim=-1)
    return attn @ V


class AttentionModule(nn.Module):
    def __init__(self, d_model, d_Q, d_K, d_V):
        super().__init__()
        self.Q = nn.Linear(d_model, d_Q, bias=False)
        self.K = nn.Linear(d_model, d_K, bias=False)
        self.V = nn.Linear(d_model, d_V, bias=False)

    def forward(self, q, k, v):
        y = attention(self.Q(q), self.K(k), self.V(v))
        return y


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, d_model, h, d_Q, d_K, d_V):
        super().__init__()
        self.linear = nn.Linear(h * d_V, d_model, bias=False)
        self.a_modules = nn.ModuleList(AttentionModule(d_model, d_Q, d_K, d_V) for _ in range(h))

    def forward(self, q, k, v):
        return self.linear(torch.cat([layer(q, k, v) for layer in self.a_modules], dim=-1))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_middle, dropout, h, d_Q, d_K, d_V):
        super().__init__()
        # multi-head_masked
        self.multi_head_masked = MultiHeadAttentionModule(d_model, h, d_Q, d_K, d_V)
        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # feed forward
        self.feed_forward = FeedForward(d_model, d_middle, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: decoder input
        """
        normed_x = self.norm1(x)
        x = x + self.dropout1(self.multi_head_masked(normed_x, normed_x, normed_x))
        normed_x = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(normed_x))

        return x


class Decoder(nn.Module):
    def __init__(self, num_blocks, d_model, d_middle, d_token, dropout, h, d_Q, d_K, d_V, use_weight_tying=False):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(d_token, d_model)
        self.layers = nn.ModuleList(
            DecoderBlock(d_model, d_middle, dropout, h, d_Q, d_K, d_V) for _ in range(num_blocks))

        self.l1 = nn.Linear(d_model, d_token)
        self.dropout = nn.Dropout(dropout)

        if use_weight_tying:
            self.embedding.weight = self.l1.weight

    def forward(self, dec_inp, pe):
        x = self.embedding(dec_inp) * np.sqrt(self.d_model)
        x = self.dropout(x + pe)
        for layer in self.layers:
            x = layer(x)
        x = self.l1(x)

        return x
