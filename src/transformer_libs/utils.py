import torch
import torch.nn as nn
import os
import time
import math
from torch.autograd import Variable
from src.transformer_libs import decoder_weight_tying
import random
import numpy as np
from src.transformer_libs import transformer
from datetime import datetime
from src.transformer_libs import model_classes_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_cuda(model):
    """Sends model from CPU to CUDA."""
    model.cuda()
    if isinstance(model, nn.Module):
        for child in model.children():
            to_cuda(child)


def optimizer_to(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def log(file_id, log_data, path, log_screen=True, log_file=True):
    file = f"log{file_id}.txt"
    if log_file:
        with open(path + file, "a") as f:
            for k, v in log_data.items():
                f.write(f'{k}:{v}\n')
                if log_screen:
                    print(f'{k}: {v}')
            print(
                "----------------------------------------------------------------------------------------------------\n")
    else:
        for k, v in log_data.items():
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


def get_pe(x):
    b, seq_len, d_model = x.shape
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    if x.is_cuda:
        pe = pe.cuda()
    return Variable(pe, requires_grad=False)


import torch
import os
import time


def save_model(model, opt, path, model_params, train_params, lr_params):
    if not os.path.exists(path):
        os.makedirs(path)

    file = os.path.join(path, f'{model_params["id"]}-{time.strftime("%Y%m%d-%H%M%S")}.pth')
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'model_params': model_params,
        'train_params': train_params,
        'lr_params': lr_params,
    }
    torch.save(save_dict, file)
    print(f"Saved model: {file}")


def load_model(path, optimizer_class=torch.optim.Adam, lr=0.001, device="cuda", training=True, model_class_name=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified model path does not exist: {path}")

    state = torch.load(path)

    model_params = state.get('model_params', {})
    if model_class_name is None:
        model_class_name = model_params.get('model_class_name', 'base_decoder')
    model_class = model_classes_dict.model_class_dict[model_class_name]
    model = model_class(**model_params)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    optimizer = None
    try:
        if 'optimizer_state_dict' in state:
            optimizer = optimizer_class(model.parameters(), lr=lr)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            optimizer_to(optimizer, device)
    except:
        print("Warning: new optimizer loaded.  Defaulting to Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_params = state.get('train_params', {})
    lr_params = state.get('lr_params', {})

    return model, optimizer, model_params, train_params, lr_params


def load_model_inference(path, class_name=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified model path does not exist: {path}")
    state = torch.load(path)
    model_params = state.get('model_params', {})
    if class_name is None:
        model_class_name = model_params.get('model_class_name', 'base_decoder')
    model_class = model_classes_dict.model_class_dict[model_class_name]
    model = model_class(**model_params)
    model.load_state_dict(state['model_state_dict'])

    return model, model_params


def default_model(model_params, train_params, lr_params, model_name='default_decoder'):
    model_class = model_classes_dict.model_class_dict[model_name]
    model_params['id'] = random.randint(0, 1_000_000)
    model_params['samples_done'] = 0
    model_params['batch_num'] = 0
    model = model_class(**model_params)
    model.cuda()
    try:
        opt = model.configure_optimizers(model_params, lr_params)
    except:
        opt = None
    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=lr_params['lr'], betas=lr_params['betas'])
    return model, opt, model_params, train_params, lr_params


def load_old_model(model_class, path, optimizer_class=torch.optim.Adam, lr=0.001, device="cuda", training=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified model path does not exist: {path}")

    state = torch.load(path)
    model_params = state.get('model_params', {})
    model_params['h'] = 12
    model_params['d_token'] = 32768
    model_params['d_k'] = 64
    model_params['d_q'] = 64
    model_params['d_v'] = 64
    model_params['num_blocks'] = 12
    model_params['d_model'] = 768
    model_params['d_middle'] = 3072
    model_params['vocab_size'] = 32768
    model_params['dropout'] = .1
    model_params['use_weight_tying'] = True
    model = model_class(**model_params)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    optimizer = None
    try:
        if 'optimizer_state_dict' in state:
            optimizer = optimizer_class(model.parameters(), lr=lr)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            optimizer_to(optimizer, device)
    except:
        print("Warning: new optimizer loaded.  Defaulting to Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_params = state.get('train_params', {})
    lr_params = state.get('lr_params', {})

    return model, optimizer, model_params, train_params, lr_params


def loss_by_position(pred, target, bs, seq_len, loss):
    loss_by_pos = [0] * seq_len
    for row1, row2 in zip(pred, target):
        pos = 0
        for seq, target in zip(row1, row2):
            loss_by_pos[pos] += (loss(seq, target).item())
            pos += 1

    loss_by_pos[:] = [x / bs for x in loss_by_pos]

    return loss_by_pos


def get_lr(lr_params):
    # lr schedule
    # batch_size factor is to compensate that smaller batches result in bigger swings
    # correspondingly lr should be smaller

    b = lr_params['batch_num']
    d_model = lr_params['d_model']
    bsf = lr_params['batch_scale_factor']  # this is a scaling factor.  I have used .1 in the past
    warmup_steps = lr_params['warmup_steps']
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


def get_cyclic_lr(lr_params):
    b = lr_params['batch_num']
    d_model = lr_params['d_model']
    stepsize = lr_params['step_size']
    min_lr = 1 / 6 * d_model ** (-.5) * (b + 1) ** (-.5)
    max_lr = d_model ** (-.5) * (b + 1) ** (-.5)
    cl = cyclical_lr(stepsize, b, min_lr, max_lr)

    warmup_steps = 4000
    warmup_lr = (b + 1) * warmup_steps ** (-1.5)

    return min(cl, warmup_lr)


def load_most_recent_model(model_dir):
    file = most_recent_file(model_dir)
    print(f'Loading most recent model: {file}')
    return load_model(file)


def print_params(params):
    for k, v in params.items():
        print(f"{k}: {v}")


def constant_lr(lr_params):
    return lr_params['constant_lr']


def create_new_model(model_class, model_params, train_params, lr_params, id=None):
    print('Generating a new model to train.')
    current_datetime = datetime.now()
    date_str = current_datetime.strftime('%Y%m%d')  # Formats as 'YYYYMMDD'
    time_str = current_datetime.strftime('%H%M%S')  # Formats as 'HHMMSS'
    model_params['id'] = id if id else f"id_{date_str}_{time_str}"
    model = model_class(**model_params)
    train_params['batch_num'] = 0
    train_params['samples_done'] = 0
    opt = torch.optim.Adam(model.parameters(), lr=lr_params['lr'])
    return model, opt, model_params, train_params, lr_params


def create_model(LOAD, directory, model_params, train_params, lr_params, file=None):
    # Set the LOAD flag to True to load either latest model or a specified model
    # Set the LOAD flag to False to generate a default model
    # if the LOAD flag is false and file is not None then a specific file is loaded
    # directory is the model directory

    if LOAD:
        if file is None:
            file = most_recent_file(directory)
            print(f'Loading model: {file}')

        # Load the model, optimizer, and model parameters from the specified file
        model, opt, model_params, train_params, lr_params = load_model(file)
    else:
        # If LOAD is set to False, generate a new model to train
        print('Generating a new model to train.')
        model, opt, model_params, train_params, lr_params = default_model(model_params, train_params, lr_params,
                                                                          model_params['model_class_name'])
        train_params['batch_num'] = 0
        train_params['samples_done'] = 0

    return model, opt, model_params, train_params, lr_params


def create_weight_tie_model(LOAD, directory, model_params, train_params, lr_params, file=None):
    # Set the LOAD flag to True to load either latest model or a specified model
    # Set the LOAD flag to False to generate a default model
    # if the LOAD flag is false and file is not None then a specific file is loaded
    # directory is the model directory

    if LOAD:
        # Initialize the file variable as None
        file = None

        # Uncomment the next line to load a specific file
        # file = config.model_directory + "model-88319-20230204-234856"

        # If no file is specified, get the most recent file in the specified directory
        if file is None:
            file = most_recent_file(directory)
            print(f'Loading model: {file}')

        # Load the model, optimizer, and model parameters from the specified file
        model, opt, model_params, train_params, lr_params = load_weight_tie_model(file, model_params, train_params,
                                                                                  lr_params)
    else:
        # If LOAD is set to False, generate a new model to train
        print('Generating a new model to train.')
        model, opt, model_params, train_params, lr_params = default_weight_tie_model(model_params, train_params,
                                                                                     lr_params)
        train_params['batch_num'] = 0
        train_params['samples_done'] = 0

    return model, opt, model_params, train_params, lr_params


def default_weight_tie_model(model_params, train_params, lr_params):
    vocab_size = model_params['vocab_size']
    num_blocks = model_params['num_blocks']
    d_model = model_params['d_model']
    d_middle = model_params['d_middle']
    dropout = model_params['dropout']
    h = model_params['h']
    d_k = model_params['d_k']
    d_q = model_params['d_q']
    d_v = model_params['d_v']
    weight_tying = model_params['weight_tying']
    model_params['id'] = random.randint(0, 1_000_000)
    model_params['samples_done'] = 0
    model_params['batch_num'] = 0
    model = decoder_weight_tying.DecoderWeightTie(num_blocks, d_model, d_middle, vocab_size, dropout, h, d_q, d_k, d_v,
                                                  weight_tying)
    model.cuda()
    opt = torch.optim.AdamW(model.parameters())
    return model, opt, model_params, train_params, lr_params


def load_weight_tie_model(file, model_params, train_params, lr_params, cuda=True):
    checkpoint = torch.load(file)

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
    if 'samples_done' in checkpoint:
        model_params['samples_done'] = checkpoint['samples_done']
    else:
        model_params['samples_done'] = 0
    if 'batch_num' in checkpoint:
        model_params['batch_num'] = checkpoint['batch_num']
    else:
        print("Model has no batch_num")
    if 'weight_decay' in checkpoint:
        model_params['weight_decay'] = checkpoint['weight_decay']
    if 'betas' in checkpoint:
        model_params['betas'] = checkpoint['betas']
    lr_params['lr'] = 2.5e-4

    model = decoder_weight_tying.DecoderWeightTie(num_blocks, d_model, d_middle, vocab_size, dropout, h, d_q, d_k, d_v,
                                                  use_weight_tying=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    if cuda:
        model.cuda(device)
    opt = torch.optim.AdamW(model.parameters())
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    if cuda:
        optimizer_to(opt, device)

    return model, opt, model_params, train_params, lr_params


def create_transformer_model(LOAD, directory, model_params, train_params, lr_params, file=None):
    # Set the LOAD flag to True to load either latest model or a specified model
    # Set the LOAD flag to False to generate a default model
    # if the LOAD flag is false and file is not None then a specific file is loaded
    # directory is the model directory

    if LOAD:
        # Initialize the file variable as None
        file = None

        # Uncomment the next line to load a specific file
        # file = config.model_directory + "model-88319-20230204-234856"

        # If no file is specified, get the most recent file in the specified directory
        if file is None:
            file = most_recent_file(directory)
            print(f'Loading model: {file}')

        # Load the model, optimizer, and model parameters from the specified file
        model, opt, model_params, train_params, lr_params = load_transformer(file, model_params, train_params,
                                                                             lr_params)
    else:
        # If LOAD is set to False, generate a new model to train
        print('Generating a new model to train.')
        model, opt, model_params, train_params, lr_params = default_transformer(model_params, train_params, lr_params)
        train_params['batch_num'] = 0
        train_params['samples_done'] = 0

    return model, opt, model_params, train_params, lr_params


def default_transformer(model_params, train_params, lr_params):
    vocab_size = model_params['vocab_size']
    num_blocks = model_params['num_blocks']
    d_model = model_params['d_model']
    d_middle = model_params['d_middle']
    dropout = model_params['dropout']
    h = model_params['h']
    d_k = model_params['d_k']
    d_q = model_params['d_q']
    d_v = model_params['d_v']
    model_params['batch_num'] = 0
    model_params['samples_done'] = 0
    model_params['id'] = random.randint(0, 1_000_000)
    model = transformer.Transformer(num_blocks, d_model, d_middle, vocab_size, dropout, h, d_q, d_k, d_v)
    model.cuda()
    opt = model.configure_optimizers(model_params, lr_params)
    return model, opt, model_params, train_params, lr_params


def load_transformer(file, model_params, train_params, lr_params, cuda=True):
    checkpoint = torch.load(file)

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
    if 'samples_done' in checkpoint:
        model_params['samples_done'] = checkpoint['samples_done']
    else:
        model_params['samples_done'] = 0
    if 'batch_num' in checkpoint:
        model_params['batch_num'] = checkpoint['batch_num']
    else:
        print("Model has no batch_num")
    if 'weight_decay' in checkpoint:
        model_params['weight_decay'] = checkpoint['weight_decay']
    if 'betas' in checkpoint:
        model_params['betas'] = checkpoint['betas']
    lr_params['lr'] = 2.5e-4

    model = transformer.Transformer(num_blocks, d_model, d_middle, vocab_size, dropout, h, d_q, d_k, d_v)

    model.load_state_dict(checkpoint['model_state_dict'])
    if cuda:
        model.cuda(device)
    opt = model.configure_optimizers(model_params, lr_params)
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    if cuda:
        optimizer_to(opt, device)

    return model, opt, model_params, train_params, lr_params
