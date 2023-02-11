import sys
# setting path
sys.path.append('../transformer_libs')
import config
import torch
import torch.nn as nn
import tokenizer
import utils
import data_utils
import datetime
import decoder
from datasets import load_dataset
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# idea is to at each stage take a fresh copy of the model
# Train it for some fixed amount of batches ~ 100
# Train the model for some lr
# Then record how well the model trained
# Then repeat the process for larger and larger learning rates
# In the end pick the lr that seems to be best



def train(model, opt, ds, tok, loss_func, bs, num_batches, seq_len, model_params, lr):
    d_model = model_params['d_model']
    batch_num = 0
    for g in opt.param_groups:
        g['lr'] = lr

    print('Starting training')
    print(f'start_batch_num: {batch_num}')
    print(f'learning rate: {opt.param_groups[0]["lr"]}')

    # params we track during training
    total_loss = 0
    total_sequences = batch_num * bs * config.accumulate_size * config.seq_len

    start_time = datetime.datetime.now()

    samples_done = model_params['samples_done']

    model.train()

    losses = []
    print(f'Training for {num_batches} batches')
    while batch_num < num_batches:
        # load sample
        combined = data_utils.get_batch(ds, tok, bs, samples_done, seq_len + 1) # combined is bs x (L + 1)
        samples_done += 1

        src = combined[:, :-1].to(device) # bs x L
        target = combined[:, 1:].to(device) # bs x L
        # positional encoding
        pe = utils.get_pe(src.size()[-1], d_model).to(device) # 1 x L x d_model
        # run through model
        pred = model(src, pe)

        # compute loss
        pred = pred.permute(0, 2, 1)
        loss = loss_func(pred, target)/config.accumulate_size

        # back prop
        loss.backward()

        # Batch is accumulated until we have enough samples
        if (samples_done + 1) % config.accumulate_size == 0:
            # opt step
            opt.step()
            opt.zero_grad()
            batch_num += 1

        total_loss += loss.item() * config.accumulate_size
        total_sequences += bs * seq_len

        # log training data
        if (samples_done + 1) % config.output_every == 0:
            end_time = datetime.datetime.now()
            lr = opt.param_groups[0]["lr"]
            current_loss = total_loss / config.output_every
            total_loss = 0
            time_for_batch_interval = end_time - start_time
            losses.append(current_loss)
            log_data = {}
            file_id = model_params['id']
            log_data['batch_num'] = batch_num
            log_data['total_seq'] = total_sequences
            log_data['lr'] = lr
            log_data['current_loss'] = current_loss
            log_data['batch_interval'] = config.output_every//config.accumulate_size
            log_data['time_for_batch_interval'] = time_for_batch_interval

            utils.log(file_id, log_data, log_screen=True, log_file=False)

            # To report loss per position uncomment both lines below
            # pred = pred.permute(0, 2, 1)
            # print(f'Loss by position: {loss_by_position(pred, target, bs, seq_len, loss)}')

            start_time = datetime.datetime.now()

    print("Ending training")
    return losses





def main():
    # Load the dataset
    LOAD_DS = True
    if LOAD_DS:
        ds = data_utils.load_book_ds(config.book_ds_file)
    else:
        ds = data_utils.create_book_ds(config.book_ds_size)
    # Load the tokenizer
    tok = tokenizer.load_tokenizer()

    # Demo the tokenizer
    tokenizer.tokenizer_test(tok)

    # Get the vocab size. +1 is due to [PAD] token
    vocab_size = tok.vocab_size + 1

    file = None
    min_lr = 1e-5
    max_lr = 1e-3
    lr = min_lr
    lr_dict = {}
    while lr < max_lr:
        # Uncomment the next line to load a specific file
        # file = config.model_directory + "model-60108-20230126-174437"

        # If no file is specified, get the most recent file in the specified directory
        if file is None:
            directory = config.model_directory
            file = utils.most_recent_file(directory)
            print(f'Loading model: {file}')

            # Load the model, optimizer, and model parameters from the specified file
            model, opt, model_params = utils.load_model(file)
        else:
            print('Generating a new model to train.')
            model, opt, model_params = utils.default_model(vocab_size)

        file = None # need to reset this so we keep loading the most recent file
        # Get the total number of parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Number of model parameters: {pytorch_total_params}')

        # Move the model and optimizer to the specified device (cuda or cpu)
        utils.to_cuda(model)
        utils.optimizer_to(opt, device)

        # Set the loss function as cross-entropy loss
        loss = nn.CrossEntropyLoss()

        # Get the batch size, number of batches, and sequence length from the config
        bs = config.batch_size
        num_batches = 100
        seq_len = config.seq_len


        # Train the model
        losses = train(model, opt, ds, tok, loss, bs, num_batches, seq_len, model_params, lr)
        lr_dict[str(lr)] = losses

        lr *= 2



    print(lr_dict)

main()