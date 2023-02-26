import config
import torch
import torch.nn as nn
from transformer_libs import tokenizer
from transformer_libs import data_utils
from transformer_libs import utils
import torch.nn.functional as F
import numpy as np
# set device to cpu so i can train at same time on the gpu without running
# out of gpu space
device = torch.device("cpu")


def prepare_eval_config():
    model_params = config.model_params
    lr_params = config.lr_params
    train_params = config.train_params
    config_params = config.config_params
    return model_params, train_params, lr_params, config_params



def main():

    # Load the tokenizer
    tok_file = config.tok_file
    tok = tokenizer.WikiTok()
    tok.load(tok_file)

    # Demo the tokenizer
    tok.tokenizer_test()

    # Load the dataset
    ds = data_utils.load_ds(config.wiki_ds_file)

    vocab_size = tok.get_vocab_size()

    model_params, train_params, lr_params, config_params = prepare_eval_config()
    model_params['vocab_size'] = vocab_size

    # To load most recent model

    file = utils.most_recent_file(config.model_directory)


    # file = None, To load a specific model uncomment this and set a file.
    model, opt, model_params, train_params, lr_params = utils.load_weight_tie_model(file, model_params, train_params, lr_params, cuda=True)
    utils.print_params(model_params)
    utils.print_params(train_params)
    utils.print_params(lr_params)
    utils.print_params(config_params)

    # Get the total number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {pytorch_total_params}')

    # Set the loss function as cross-entropy loss
    loss_func = nn.CrossEntropyLoss()
    lr_func = utils.constant_lr
    dl_params = {'L': model_params['seq_len'], 'tok': tok, 'ds': ds}
    dl = data_utils.WikipediaDL(model_params['bs'], model_params['samples_done'], dl_params)

    num_batches = 1_000

    accumulated_stats = eval(num_batches, model, model_params, dl)


def eval(num_batches, model, model_params, dl):
    model.eval()
    start_batch_num = model_params['batch_num']
    batch_num = start_batch_num


    samples_done = model_params['samples_done']
    print(f'samples_done: {samples_done}')

    print('Starting eval')
    print(f'start_batch_num: {batch_num}')

    accumulated_stats = dict()

    while batch_num < start_batch_num + num_batches:
        # update dl parameters
        update_params = {'batch_num': batch_num, 'samples_done': samples_done, }
        dl.update(update_params)
        # load sample
        src, target = dl()
        target = target.to(device)
        samples_done += 1

        # run through model
        pred = model(src)

        # different loss stuff
        accumulated_stats = compute_pred_stats(pred, target, accumulated_stats)



    return accumulated_stats

def compute_pred_stats(pred, target, accumulated_stats):
    pred = F.softmax(pred)
    loss = nn.CrossEntropyLoss(torch.permute(pred,(0,2,1)))
    perplexity = np.exp(loss)
    return accumulated_stats