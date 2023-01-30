import torch
import torch.nn as nn
import os
import time
import math
import config
from torch.autograd import Variable
import decoder
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_cuda(model):
    """Sends model from CPU to CUDA."""
    model.cuda()
    if isinstance(model, nn.Module):
        for child in model.children():
            to_cuda(child)


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def log(file_id, log_data, log_screen=True):
    path = config.log_directory
    file = f"log{file_id}.txt"
    with open(path + file, "a") as f:
        for k, v in log_data.items():
            f.write(f'{k}:{v}\n')
            if log_screen:
                print(f'{k}: {v}')
        print("----------------------------------------------------------------------------------------------------\n")


def most_recent_file(directory):
    files = os.listdir(directory)

    latest_file = ''
    latest_time = 0

    for file in files:
        path = os.path.join(directory, file)
        modification_time = os.path.getmtime(path)
        if modification_time > latest_time:
            latest_file = path
            latest_time = modification_time

    return latest_file


def get_mask(size):
    # Create a (size, size) tensor filled with ones
    mask = torch.ones((size, size))

    # Set the upper triangular elements to 0
    mask = torch.triu(mask, diagonal=1)

    # Convert the tensor to a boolean tensor
    mask = mask.bool()

    # Add an additional dimension of size 1 to the start of the tensor
    mask = mask.unsqueeze(0)
    return torch.logical_not(mask)


def get_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return Variable(pe, requires_grad=False)


def load_model(file):
    checkpoint = torch.load(file)
    model_params = {}
    num_blocks = model_params['num_blocks'] = checkpoint['num_blocks']
    d_model = model_params['d_model'] = checkpoint['d_model']
    d_middle = model_params['d_middle'] = checkpoint['d_middle']
    vocab_size = model_params['vocab_size'] = checkpoint['vocab_size']
    dropout = model_params['dropout'] = checkpoint['dropout']
    h = model_params['h'] = checkpoint['h']
    d_q = model_params['d_q'] = checkpoint['d_q']
    d_k = model_params['d_k'] = checkpoint['d_k']
    d_v = model_params['d_v'] = checkpoint['d_v']

    model_params['id'] = checkpoint['id']

    model_params['start_batch_num'] = checkpoint['start_batch_num']

    model = decoder.Decoder(num_blocks, d_model, d_middle, vocab_size, dropout, h, d_q, d_k, d_v, use_weight_tying=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, opt, model_params


def default_model(vocab_size):
    model_params = {}
    num_blocks = model_params['num_blocks'] = config.model_params['num_blocks']
    d_model = model_params['d_model'] = config.model_params['d_model']
    d_middle = model_params['d_middle'] = config.model_params['d_middle']
    dropout = model_params['dropout'] = config.model_params['dropout']
    h = model_params['h'] = config.model_params['h']
    d_k = model_params['d_k'] = config.model_params['d_k']
    d_q = model_params['d_q'] = config.model_params['d_q']
    d_v = model_params['d_v'] = config.model_params['d_v']
    model_params['vocab_size'] = vocab_size
    model_params['start_batch_num'] = 0
    model_params['id'] = random.randint(0, 100000)
    model = decoder.Decoder(num_blocks, d_model, d_middle, vocab_size, dropout, h, d_q, d_k, d_v, use_weight_tying=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-4, betas=(0.9, 0.98), eps=1e-9)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model, opt, model_params


def save_model(model, opt, model_params):
    path = f'.\models\model-{model_params["id"]}-' + time.strftime("%Y%m%d-%H%M%S")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'num_blocks': model_params['num_blocks'],
        'd_model': model_params['d_model'],
        'd_middle': model_params['d_middle'],
        'vocab_size': model_params['vocab_size'],
        'dropout': model_params['dropout'],
        'h': model_params['h'],
        'd_q': model_params['d_q'],
        'd_k': model_params['d_k'],
        'd_v': model_params['d_v'],
        'start_batch_num': model_params['start_batch_num'],
        'id': model_params['id'],
    }, path)

    print(f"Saving model: {path}")


def loss_by_position(pred, target, bs, seq_len, loss):
    loss_by_pos = [0] * seq_len
    for row1, row2 in zip(pred, target):
        pos = 0
        for seq, target in zip(row1, row2):
            loss_by_pos[pos] += (loss(seq, target).item())
            pos += 1

    loss_by_pos[:] = [x / bs for x in loss_by_pos]

    return loss_by_pos


def get_lr(b, d_model):
    # lr schedule
    # batch_size factor is to compensate that smaller batches result in bigger swings
    # correspondigly lr should be smaller

    bsf = config.batch_scale_factor

    warmup_steps = 4000
    lr = bsf * d_model ** (-.5) * min((b + 1) ** (-.5), (b + 1) * warmup_steps ** (-1.5))
    return lr

# This implements cyclic lr
def relative(batch_num, stepsize, scaler):
    cycle = np.floor(1 + batch_num / (2 * stepsize))
    x = abs(batch_num / stepsize - 2 * cycle + 1)
    return max(0, (1 - x)) * scaler

def cyclical_lr(stepsize, batch_num, min_lr=2.e-5, max_lr=2.5e-4):
    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = 1
    return min_lr + (max_lr - min_lr) * relative(batch_num, stepsize, scaler)


def get_cyclic_lr(b, d_model, stepsize):
    min_lr = 1/6 * d_model ** (-.5) * (b + 1) ** (-.5)
    max_lr = d_model ** (-.5) * (b + 1) ** (-.5)
    cl = cyclical_lr(stepsize, b, min_lr, max_lr)

    warmup_steps = 4000
    warmup_lr = (b + 1) * warmup_steps ** (-1.5)

    return min(cl, warmup_lr)


