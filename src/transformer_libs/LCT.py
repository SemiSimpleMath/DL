import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.transformer_libs import utils


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


# This was my original implementation, however, it is possible to ditch this module
# and apply attention to all the heads in parallel. See new implementation below
class AttentionModule_old(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, dropout):
        super().__init__()
        self.Q = nn.Linear(d_model, d_q, bias=False)
        self.K = nn.Linear(d_model, d_k, bias=False)
        self.V = nn.Linear(d_model, d_v, bias=False)
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

# the old way of doing MHAM
class MultiHeadAttentionModule_old(nn.Module):
    def __init__(self, d_model, h, d_q, d_k, d_v, dropout):
        super().__init__()
        self.l_proj = nn.Linear(h * d_v, d_model, bias=False)
        self.a_modules = nn.ModuleList(AttentionModule_old(d_model, d_q, d_k, d_v, dropout) for _ in range(h))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # collect all the heads together and project. Each head has shape (b, L, d_v)
        # the concatenated shape is (b, L, h * d_v)
        # This is projected to (b, L, d_v) This is the end result of the MHAM.
        return self.dropout(self.l_proj(torch.cat([layer(q, k, v) for layer in self.a_modules], dim=-1)))


class MultiHeadAttentionModule(nn.Module):
    def __init__(self, d_model, h, d_q, d_k, d_v, dropout):
        super().__init__()
        self.combined_qkv = nn.Linear(d_model, 3*d_model)
        self.l_proj = nn.Linear(h * d_v, d_model)
        self.a_modules = nn.ModuleList(AttentionModule_old(d_model, d_q, d_k, d_v, dropout) for _ in range(h))
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


class Decoder_context_head(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_blocks = kwargs.get('num_blocks')
        self.d_model = kwargs.get('d_model')
        self.d_middle = kwargs.get('d_middle')
        self.d_token = kwargs.get('d_token', kwargs.get('vocab_size'))
        self.d_vocab = kwargs.get('vocab_size')
        self.dropout = kwargs.get('dropout')
        self.h = kwargs.get('h')
        self.d_q = kwargs.get('d_q')
        self.d_k = kwargs.get('d_k')
        self.d_v = kwargs.get('d_v')

        self.embedding = nn.Embedding(self.d_token, self.d_model)
        self.layers = nn.ModuleList(
            DecoderBlock(self.d_model, self.d_middle, self.dropout, self.h, self.d_q, self.d_k, self.d_v) for _ in range(self.num_blocks))

        self.dropout = nn.Dropout(self.dropout)
        self.ln_out = nn.LayerNorm(self.d_model)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / np.sqrt(2 * self.num_blocks))

    def forward(self, dec_inp):
        x = self.embedding(dec_inp) * np.sqrt(self.d_model)  # bs x L x d_model
        pe = utils.get_pe(x)
        x = self.dropout(x + pe)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        return x


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
                # Check if this parameter is tied and handle accordingly

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

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


class Decoder_tail(nn.Module):
    def __init__(self, **kwargs):
        self.num_blocks = kwargs.get('num_blocks')
        self.d_model = kwargs.get('d_model')
        self.d_middle = kwargs.get('d_middle')
        self.d_token = kwargs.get('d_token', kwargs.get('vocab_size'))
        self.d_vocab = kwargs.get('vocab_size')
        self.dropout = kwargs.get('dropout')
        self.h = kwargs.get('h')
        self.d_q = kwargs.get('d_q')
        self.d_k = kwargs.get('d_k')
        self.d_v = kwargs.get('d_v')


        super().__init__()
        self.layers = nn.ModuleList(
            DecoderBlock(self.d_model, self.d_middle, self.dropout, self.h, self.d_q, self.d_k, self.d_v) for _ in range(self.num_blocks))

        self.l_out = nn.Linear(self.d_model, self.d_token, bias=False)
        self.dropout = nn.Dropout(self.dropout)
        self.ln_out = nn.LayerNorm(self.d_model)
        self.embedding = nn.Embedding(self.d_token, self.d_model)
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / np.sqrt(2 * self.num_blocks))

    def forward(self, context, dec_inp):
        x = self.embedding(dec_inp) * np.sqrt(self.d_model)  # bs x L x d_model
        result = torch.cat([context, x], dim=1) # L->L+1
        pe = utils.get_pe(result)
        x = self.dropout(result + pe)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        x = self.l_out(x)
        return x[:,1:,:]

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
                # Check if this parameter is tied and handle accordingly

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
            no_decay.add('l_out.weight')
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

class TransitionModule(nn.Module):
    def __init__(self, seq_len, r, d_model):
        super().__init__()
        # Linear layer to transform the sequence length
        self.transform = nn.Linear(seq_len, r)

    def forward(self, x):
        bs, seq_len, d_model = x.size()  # batch size, sequence length, d_model
        # Reshape to (bs, d_model, seq_len) for linear layer
        x = x.transpose(1, 2)
        # Apply linear transformation
        x = self.transform(x)
        # Reshape back to (bs, r, d_model)
        x = x.transpose(1, 2)
        return x

class ReverseTransitionModule(nn.Module):
    def __init__(self, r, seq_len, d_model):
        super().__init__()
        # Linear layer to transform the compressed sequence length back to original
        self.expand = nn.Linear(r, seq_len)

    def forward(self, x):
        bs, r, d_model = x.size()  # batch size, compressed sequence length, d_model
        # Reshape to (bs, d_model, r) for the linear layer
        x = x.transpose(1, 2)
        # Apply linear transformation to expand
        x = self.expand(x)
        # Reshape back to (bs, seq_len, d_model)
        x = x.transpose(1, 2)
        return x

class LongContextTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.head_params = kwargs['head']
        self.tail_params = kwargs['tail']
        r = self.head_params['r']
        seq_len = self.head_params['seq_len']
        d_model = self.tail_params['d_model']
        self.head = Decoder_context_head(**self.head_params)
        self.transition = TransitionModule(seq_len, r, d_model)
        self.tail = Decoder_tail(**self.tail_params)

    def configure_optimizers(self, model_params, lr_params):
        head_params = model_params['head']
        tail_params = model_params['tail']
        self.head.configure_optimizers(head_params, lr_params)
        self.tail.configure_optimizers(tail_params, lr_params)
    def forward(self, x):
        context_input, seq_input = x
        head_output = self.head(context_input)
        transition_output = self.transition(head_output)
        tail_output = self.tail(transition_output, seq_input)

        return tail_output
