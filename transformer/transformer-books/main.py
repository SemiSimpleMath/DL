import config
import torch
import torch.nn as nn
from transformer_libs import tokenizer
from transformer_libs import data_utils
from transformer_libs import utils
from transformer_libs import train
import os
import random

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_train_config():
    model_params = config.model_params
    lr_params = config.lr_params
    train_params = config.train_params
    config_params = config.config_params
    return model_params, train_params, lr_params, config_params


def main():
    torch.manual_seed(0)

    # Load the dataset
    ds = data_utils.load_ds(config.book_ds_file)
    random.shuffle(ds)
    # Load the tokenizer
    tok_file = config.tok_file
    tok = tokenizer.load_tokenizer(tok_file)

    # Demo the tokenizer
    tokenizer.tokenizer_test(tok)

    # Get the vocab size. +1 is due to [PAD] token which we force to be the last token
    vocab_size = tok.vocab_size + 1

    model_params, train_params, lr_params, config_params = prepare_train_config()
    model_params['vocab_size'] = vocab_size

    # To load most recent model set LOAD flag to True
    LOAD = True
    # file = None To load a specific model uncomment this and set a file.
    model, opt, model_params, train_params, lr_params = utils.create_model(LOAD, config_params['model_directory'],
                                                                           model_params, train_params, lr_params,
                                                                           file=None)

    utils.print_params(model_params)
    utils.print_params(train_params)
    utils.print_params(lr_params)
    utils.print_params(config_params)

    # Get the total number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {pytorch_total_params}')

    # Set the loss function as cross-entropy loss
    loss_func = nn.CrossEntropyLoss()
    dl_func = data_utils.get_batch
    lr_func = utils.constant_lr

    trainer = train.Train(model, opt, model_params, train_params, lr_params, config_params, tok, ds, loss_func, dl_func,
                          lr_func)
    # eval_model(model, tok, loss, bs, 100, seq_len, model_params)

    trainer.train(50_000)


main()



