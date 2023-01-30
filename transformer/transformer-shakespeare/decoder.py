import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_upper_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        create_upper_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def positional_encoding(seq_len, d_model):
    max_len = seq_len + 1
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


# feed forward

class FeedForward(nn.Module):
    def __init__(self, d_model, d_middle):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_middle)
        self.l2 = nn.Linear(d_middle, d_model)

    def forward(self, x):
        x = self.l2(F.relu(self.l1(x)))

        return x


class AttentionModule(nn.Module):
    def __init__(self, d_model, d_Q, d_K, d_V):
        super().__init__()
        self.Q = nn.Linear(d_model, d_Q, bias=False)
        self.K = nn.Linear(d_model, d_K, bias=False)
        self.V = nn.Linear(d_model, d_V, bias=False)

    def forward(self, q, k, v, mask=None):
        y = self.attention(self.Q(q), self.K(k), self.V(v), mask)
        return y

    def attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = attn @ V
        return attn


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, d_model, h, d_Q, d_K, d_V):
        super().__init__()
        self.linear = nn.Linear(h * d_V, d_model, bias=False)
        self.a_modules = nn.ModuleList(AttentionModule(d_model, d_Q, d_K, d_V) for _ in range(h))

    def forward(self, q, k, v, mask=None):
        combines = []

        for layer in self.a_modules:
            y = layer(q, k, v, mask)

            combines.append(y)

        y = torch.cat(combines, -1)

        y = self.linear(y)

        return y


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_middle, dropout, h, d_Q, d_K, d_V):
        super().__init__()
        # multihead_masked
        self.multi_head_masked = MultiHeadAttentionModule(d_model, h, d_Q, d_K, d_V)
        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # feed forward
        self.feed_forward = FeedForward(d_model, d_middle)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x: decoder input
        """
        x = x + self.dropout(self.multi_head_masked(x, x, x, mask))
        x = self.norm1(x)
        x = x + self.dropout(self.feed_forward(x))
        x = self.norm2(x)
        return x



class Decoder(nn.Module):
    def __init__(self, num_blocks, d_model, d_middle, d_token, dropout, h, d_Q, d_K, d_V):
        super().__init__()
        self.embedding = nn.Embedding(d_token, d_model)
        self.layers = nn.ModuleList(
            DecoderBlock(d_model, d_middle, dropout, h, d_Q, d_K, d_V) for _ in range(num_blocks))

        self.l1 = nn.Linear(d_model, d_token)

    def forward(self, dec_inp, pe, dec_mask=None):
        x = self.embedding(dec_inp)
        x = x + pe
        for layer in self.layers:
            x = layer(x, dec_mask)
        x = self.l1(x)

        return x


MAX_MASK_SIZE = 500
mask = create_upper_mask(MAX_MASK_SIZE)
def get_mask(size):
    msk = mask[0, :size, : size]
    return msk

MAX_SEQ_LEN = 500
MAX_D_MODEL = 512
pos_enc = positional_encoding(MAX_SEQ_LEN, MAX_D_MODEL)
def get_pe(seq_len, d_model):
    pe = pos_enc[0, :seq_len, :d_model]
    return pe