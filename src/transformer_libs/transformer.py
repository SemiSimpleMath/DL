import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer_libs import utils

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

    def attention(self, Q, K, V, mask=False):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
        if mask:
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

    def forward(self, q, k, v, mask):
        # collect all the heads together and project. Each head has shape (b, L, d_v)
        # the concatenated shape is (b, L, h * d_v)
        # This is projected to (b, L, d_v) This is the end result of the MHAM.
        return self.dropout(self.l_proj(torch.cat([layer(q, k, v, mask) for layer in self.a_modules], dim=-1)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_middle, dropout, h, d_Q, d_K, d_V):
        super().__init__()

        # multihead
        self.multi_head = MultiHeadAttentionModule(d_model, h, d_Q, d_K, d_V, dropout)
        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # feed forward
        self.feed_forward = FeedForward(d_model, d_middle, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=False):
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

    def forward(self, x):
        pe = utils.get_pe(x)
        pe = pe.to(device)
        x += pe
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_middle, dropout, h, d_Q, d_K, d_V):
        super().__init__()

        # multihead_masked
        self.multi_head_masked = MultiHeadAttentionModule(d_model, h, d_Q, d_K, d_V, dropout)
        # multihead_encoder
        self.multi_head_encoder = MultiHeadAttentionModule(d_model, h, d_Q, d_K, d_V, dropout)
        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # feed forward
        self.feed_forward = FeedForward(d_model, d_middle, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        """
        x: decoder input
        y: encoder output
        """
        x = x + self.dropout(self.multi_head_masked(x, x, x, True))
        x = self.norm1(x)
        x = x + self.dropout(self.multi_head_encoder(x, y, y, False))
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


    def forward(self, enc_out, dec_inp):
        pe = utils.get_pe(dec_inp)
        pe = pe.to(device)
        x = dec_inp + pe
        y = enc_out
        for layer in self.layers:
            x = layer(x, y)
        x = self.l1(x)

        return x


class Transformer(nn.Module):
    def __init__(self, num_blocks, d_model, d_middle, d_token, dropout, h, d_Q, d_K, d_V):
        super().__init__()
        self.encoder = Encoder(num_blocks, d_model, d_middle, dropout, h, d_Q, d_K, d_V)
        self.decoder = Decoder(num_blocks, d_model, d_middle, d_token, dropout, h, d_Q, d_K, d_V)
        self.embedding_enc = nn.Embedding(d_token, d_model)
        self.embedding_dec = nn.Embedding(d_token, d_model)
        self.embedding_enc.weight = self.embedding_dec.weight
        self.d_model = d_model

    def forward(self, src):
        enc_src, dec_src = src
        enc_src = enc_src.to(device)
        dec_src = dec_src.to(device)
        dec_in = self.embedding_dec(dec_src) * np.sqrt(self.d_model)  # bs x L x d_model
        enc_in = self.embedding_enc(enc_src) * np.sqrt(self.d_model)  # bs x L x d_model
        return self.decoder(self.encoder(enc_in), dec_in)

    # This is from Andre Karpathy.  I did originally something much simpler, but I am humble enough
    # to learn from the best.
    def configure_optimizers(self, model_params, lr_params):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        if 'l_out.weight' in decay:
            decay.remove('l_out.weight')
        if 'embedding_dec.weight' in decay:
            decay.remove('embedding_dec.weight')
        if 'embedding_dec.weight' in no_decay:
            no_decay.remove('embedding_dec.weight')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": model_params['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr_params['lr'], betas=model_params['betas'])
        return optimizer

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
