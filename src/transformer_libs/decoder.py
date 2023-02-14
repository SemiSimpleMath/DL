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

    def attention(self, Q, K, V):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
        scores += torch.triu(torch.ones_like(scores) * float("-inf"), diagonal=1)  # mask subsequent positions
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return attn @ V

    def forward(self, q, k, v):
        y = self.attention(self.Q(q), self.K(k), self.V(v))
        return y


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


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_middle, dropout, h, d_Q, d_K, d_V):
        super().__init__()
        # multi-head_masked
        self.multi_head_masked = MultiHeadAttentionModule(d_model, h, d_Q, d_K, d_V, dropout)
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
        x = x + self.multi_head_masked(normed_x, normed_x, normed_x)
        normed_x = self.norm2(x)
        x = x + self.feed_forward(normed_x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_blocks, d_model, d_middle, d_token, dropout, h, d_Q, d_K, d_V, use_weight_tying=False):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(d_token, d_model)
        self.layers = nn.ModuleList(
            DecoderBlock(d_model, d_middle, dropout, h, d_Q, d_K, d_V) for _ in range(num_blocks))

        self.l_out = nn.Linear(d_model, d_token, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ln_out = nn.LayerNorm(d_model)

        if use_weight_tying:
            self.l_out.weight = self.embedding.weight

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / np.sqrt(2 * num_blocks))

    def forward(self, dec_inp, pe):
        x = self.embedding(dec_inp) * np.sqrt(self.d_model)  # bs x L x d_model
        x = self.dropout(x + pe)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        return self.l_out(x)

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
