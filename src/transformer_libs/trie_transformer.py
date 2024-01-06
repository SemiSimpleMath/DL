import src.transformer_libs.utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))



class FeedForward(nn.Module):
    def __init__(self, d_model, d_middle, dropout):
        super().__init__()
        self.l_feed = nn.Linear(d_model, d_middle)
        self.l_proj = nn.Linear(d_middle, d_model)
        self.dropout = nn.Dropout(dropout)
        self.NG = NewGELU()

    def forward(self, x):
        return self.dropout(self.l_proj(self.NG(self.l_feed(x))))


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, d_model, h, d_q, d_k, d_v, dropout):
        super().__init__()
        self.combined_qkv = nn.Linear(d_model, 3*d_model)
        self.l_proj = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.h = h

    def forward(self, x):
        bs, L, d_model = x.size()  # batch size, sequence length, d_model
        Q, K, V = self.combined_qkv(x).split(d_model, dim=2)
        d_k = K.size(-1)

        K = K.view(bs, L, self.h, d_model // self.h).transpose(1, 2)
        Q = Q.view(bs, L, self.h, d_model // self.h).transpose(1, 2)
        V = V.view(bs, L, self.h, d_model // self.h).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
        scores += torch.triu(torch.ones_like(scores) * float("-inf"), diagonal=1)  # mask subsequent positions
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn = attn @ V
        attn = attn.transpose(1, 2).contiguous().view(bs, L, d_model)  # collect all heads together
        # output projection
        attn = self.dropout(self.l_proj(attn))
        return attn


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_middle, dropout, h, d_q, d_k, d_v):
        super().__init__()
        # multi-head_masked
        self.multi_head_masked = MultiHeadAttentionModule(d_model, h, d_q, d_k, d_v, dropout)
        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # feed forward
        self.feed_forward = FeedForward(d_model, d_middle, dropout)

    def forward(self, x):
        """
        x: decoder input
        """
        normed_x = self.norm1(x)
        x = x + self.multi_head_masked(normed_x)
        normed_x = self.norm2(x)
        x = x + self.feed_forward(normed_x)

        return x


class TrieTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_embd_layers = kwargs['num_embd_layers']
        self.base_embd_size = kwargs['base_embd_size']
        self.num_blocks_embed = kwargs['num_blocks_embed']
        self.embed_heads = kwargs['embed_heads']
        self.d_token = kwargs.get('d_token', kwargs.get('vocab_size'))
        self.dropout_rate = kwargs.get('dropout')
        self.tail_heads = kwargs['tail_heads']
        self.num_tail_blocks = kwargs['num_tail_blocks']

        # Common setup for each layer set
        def setup_layers(embd_size, layer_size, num_blocks, h):
            d_qk = layer_size // h
            embd = nn.Embedding(self.d_token, embd_size)
            layers = nn.ModuleList(
                DecoderBlock(layer_size, 4 * layer_size, self.dropout_rate, h, d_qk, d_qk, d_qk)
                for _ in range(num_blocks)
            )
            return embd, layers

        self.embd_layers = nn.ModuleList()
        self.layer_groups = nn.ModuleList()
        total_embd_size = 0

        for i in range(self.num_embd_layers):
            current_embd_size = self.base_embd_size
            total_embd_size = self.base_embd_size * (i + 1)
            embd, layers = setup_layers(current_embd_size, total_embd_size, self.num_blocks_embed, self.embed_heads)
            self.embd_layers.append(embd)
            self.layer_groups.append(layers)

        self.size_tail = total_embd_size
        self.tail_layers = nn.ModuleList(
            DecoderBlock(self.size_tail, 4 * self.size_tail, self.dropout_rate, self.tail_heads,
                         self.size_tail // self.tail_heads,
                         self.size_tail // self.tail_heads, self.size_tail // self.tail_heads)
            for _ in range(self.num_tail_blocks)
        )

        self.l_out = nn.Linear(self.size_tail, self.d_token, bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.ln_out = nn.LayerNorm(total_embd_size)

    def forward(self, dec_inp):
        def process_layers(embd, layers, inp, prev_output=None):
            x = embd(inp) * np.sqrt(embd.embedding_dim)
            if prev_output is not None:
                x = torch.cat([prev_output, x], dim=-1)
            pe = utils.get_pe(x)
            x = self.dropout(x + pe)
            for layer in layers:
                x = layer(x)
            return x

        x = None
        for embd, layers in zip(self.embd_layers, self.layer_groups):
            x = process_layers(embd, layers, dec_inp, x)

        tail_x = x
        for layer in self.tail_layers:
            tail_x = layer(tail_x)
        tail_x = self.ln_out(tail_x)
        tail_x = self.l_out(tail_x)
        return tail_x
