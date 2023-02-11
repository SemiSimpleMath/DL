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

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(model, tok, loss_func, bs, num_batches, seq_len, model_params):
    print("Starting eval")
    model.eval()

    ds = data_utils.load_book_ds(config.valid_ds_file)

    batch_num = 0
    d_model = model_params['d_model']
    samples_done = 0
    total_loss = 0
    start_time = datetime.datetime.now()

    while batch_num < num_batches:
        # load sample
        combined = data_utils.get_batch(ds, tok, bs, samples_done, seq_len + 1)  # combined is bs x (L + 1)
        samples_done += 1

        src = combined[:, :-1].to(device)  # bs x L
        target = combined[:, 1:].to(device)  # bs x L
        # positional encoding
        pe = utils.get_pe(src.size()[-1], d_model).to(device)  # 1 x L x d_model
        # run through model
        pred = model(src, pe)
        # compute loss
        pred = pred.permute(0, 2, 1)
        loss = loss_func(pred, target)

        total_loss += loss.item()

        if (samples_done + 1) % config.accumulate_size == 0:
            batch_num += 1

        # log eval data
        if (samples_done + 1) % config.output_every == 0:
            end_time = datetime.datetime.now()
            current_loss = total_loss / config.output_every
            total_loss = 0
            time_for_batch_interval = end_time - start_time

            log_data = {}
            file_id = model_params['id']
            log_data['batch_num'] = batch_num
            log_data['current_loss'] = current_loss
            log_data['batch_interval'] = config.output_every // config.accumulate_size
            log_data['time_for_batch_interval'] = time_for_batch_interval

            utils.log('eval' + str(file_id), log_data, log_screen=True)

            # To report loss per position uncomment both lines below
            # pred = pred.permute(0, 2, 1)
            # print(f'Loss by position: {loss_by_position(pred, target, bs, seq_len, loss)}')

            start_time = datetime.datetime.now()

    print("Ending eval")
    return


def train(model, opt, ds, tok, loss_func, bs, num_batches, seq_len, model_params):
    model.train()
    d_model = model_params['d_model']
    start_batch_num = model_params['batch_num']
    batch_num = start_batch_num
    lr_scale_factor = .1
    lr = lr_scale_factor * utils.get_cyclic_lr(batch_num, d_model, 1000)
    for g in opt.param_groups:
        g['lr'] = lr

    print('Starting training')
    print(f'start_batch_num: {batch_num}')
    print(f'learning rate: {opt.param_groups[0]["lr"]}')

    # params we track during training
    total_loss = 0
    total_sequences = batch_num * bs * config.accumulate_size * config.seq_len

    start_time = datetime.datetime.now()

    samples_done = 0 #model_params['samples_done']
    print(f'samples_done: {samples_done}')
    accumulation_counter = 0
    while batch_num < start_batch_num + num_batches:
        # load sample
        combined = data_utils.get_chat_batch(ds, bs, samples_done)  # combined is bs x (L + 1)
        samples_done += 1

        src = combined[:, :-1].to(device)  # bs x L
        target = combined[:, 1:].to(device)  # bs x L
        # positional encoding
        pe = utils.get_pe(src.size()[-1], d_model).to(device)  # 1 x L x d_model
        # run through model
        pred = model(src, pe)
        # compute loss
        pred = pred.permute(0, 2, 1)
        loss = loss_func(pred, target)/config.accumulate_size
        loss.backward()

        # Batch is accumulated until we have enough samples
        if (accumulation_counter + 1) % config.accumulate_size == 0:
            # opt step
            opt.step()
            opt.zero_grad()
            batch_num += 1

        total_loss += loss.item() * config.accumulate_size
        total_sequences += bs * seq_len

        # log training data
        if (accumulation_counter + 1) % config.output_every == 0:
            end_time = datetime.datetime.now()
            lr = opt.param_groups[0]["lr"]
            current_loss = total_loss / config.output_every
            total_loss = 0
            time_for_batch_interval = end_time - start_time

            log_data = {}
            file_id = model_params['id']
            log_data['batch_num'] = batch_num
            log_data['total_seq'] = total_sequences
            log_data['lr'] = lr
            log_data['current_loss'] = current_loss
            log_data['accumulate_size'] = config.accumulate_size
            log_data['output_every'] = config.output_every // config.accumulate_size
            log_data['time_for_batch_interval'] = time_for_batch_interval

            utils.log(file_id, log_data, log_screen=True)

            # To report loss per position uncomment both lines below
            # pred = pred.permute(0, 2, 1)
            # print(f'Loss by position: {loss_by_position(pred, target, bs, seq_len, loss)}')

            start_time = datetime.datetime.now()

        # update learning rate
        if (accumulation_counter + 1) % config.update_lr_every == 0:
            lr = lr_scale_factor * utils.get_cyclic_lr(batch_num, d_model, 10000)
            for g in opt.param_groups:
                g['lr'] = lr

        # save the model
        if (accumulation_counter + 1) % config.save_every == 0:
            model_params['batch_num'] = batch_num
            model_params['samples_done'] = samples_done
            utils.save_model(model, opt, model_params)
            print(f'Epoch progress: {samples_done * bs / len(ds)}')

        # if (accumulation_counter + 1) % config.eval_every == 0:
        #     model_params['batch_num'] = batch_num
        #     eval_model(model, tok, loss_func, bs, 100, seq_len, model_params)
        #     model.train()
        #     print("Starting re-starting training")

        accumulation_counter += 1
    print("Ending training")
    utils.save_model(model, opt, model_params)
    return


def main():
    torch.manual_seed(0)

    # Load the dataset
    ds = data_utils.load_chat_ds(config.chat_ds_file)
    tok = tokenizer.load_tokenizer()

    # Demo the tokenizer
    tokenizer.tokenizer_test(tok)

    # Get the vocab size. +1 is due to [PAD] token. For whatever reason adding tokens after training
    # does not grow vocab size.  So we need to add +1 here manually

    vocab_size = tok.vocab_size + 1

    # Set the LOAD flag to True to load either latest model or a specified model
    # Set the LOAD flag to False to generate a default model
    LOAD = True

    if LOAD:
        # Initialize the file variable as None
        file = config.model_directory + "model-88319-20230204-234856"

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
        # If LOAD is set to False, generate a new model to train
        print('Generating a new model to train.')
        model, opt, model_params = utils.default_model(vocab_size)
        model_params['batch_num'] = 0

    # Get the total number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {pytorch_total_params}')

    # Set the loss function as cross-entropy loss
    loss = nn.CrossEntropyLoss()

    # Get the batch size, number of batches, and sequence length from the config
    bs = config.batch_size
    num_batches = config.num_batches
    seq_len = config.seq_len

    #eval_model(model, tok, loss, bs, 100, seq_len, model_params)
    # Train the model
    train(model, opt, ds, tok, loss, bs, num_batches, seq_len, model_params)

    # Save the model, optimizer, and model parameters
    # utils.save_model(model, opt, model_params)


main()

# tok = tokenizer.load_tokenizer()
# # # ds = data_utils.load_chat_ds(config.chat_ds_file)
# ds = data_utils.create_chat_data(tok)
# data_utils.save_chat_ds(ds, config.chat_ds_file)
