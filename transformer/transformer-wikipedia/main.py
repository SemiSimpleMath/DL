import torch
import torch.nn as nn
import tokenizer
import utils
import data_utils
import datetime
import config
from datasets import load_dataset
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, opt, ds, tok, loss_func, bs, num_batches, seq_len, model_params):
    d_model = model_params['d_model']
    start_batch_num = model_params['start_batch_num']

    lr = utils.get_lr(start_batch_num, d_model)
    for g in opt.param_groups:
        g['lr'] = lr

    print('Starting training')
    print(f'start_batch_num: {start_batch_num}')
    print(f'learning rate: {opt.param_groups[0]["lr"]}')

    # params we track during training
    total_loss = 0
    total_sequences = start_batch_num * bs

    start_time = datetime.datetime.now()

    #positional encoding
    pe = utils.get_pe(seq_len, d_model).to(device)

    for batch_num in range(start_batch_num, start_batch_num + num_batches):

        opt.zero_grad()

        # load batch
        combined = data_utils.get_batch(ds, tok, bs, batch_num, seq_len + 1)
        src = combined[:, :-1].to(device)
        target = combined[:, 1:].to(device)

        # run through model
        pred = model(src, pe)

        # compute loss
        pred = pred.permute(0, 2, 1)
        loss = loss_func(pred, target)

        # back prop
        loss.backward()

        # opt step
        opt.step()

        total_loss += loss.item()
        total_sequences += bs * seq_len

        # log training data
        if batch_num % config.output_every == 0 and batch_num != start_batch_num:
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
            log_data['batch_interval'] = config.output_every
            log_data['time_for_batch_interval'] = time_for_batch_interval

            utils.log(file_id, log_data, log_screen=True)

            # To report loss per position uncomment both lines below
            # pred = pred.permute(0, 2, 1)
            # print(f'Loss by position: {loss_by_position(pred, target, bs, seq_len, loss)}')

            start_time = datetime.datetime.now()

        if batch_num % config.lr_step == 0 and batch_num != start_batch_num:
            lr = utils.get_cyclic_lr(batch_num, d_model, stepsize=400)
            for g in opt.param_groups:
                g['lr'] = lr

        # save the model
        if batch_num % config.save_every == 0 and batch_num != start_batch_num:
            model_params['start_batch_num'] = batch_num + 1
            utils.save_model(model, opt, model_params)


def main():
    # Load the dataset
    ds = load_dataset(config.ds_path, config.ds_file)

    # Load the tokenizer
    tok = tokenizer.load_tokenizer()

    # Demo the tokenizer
    tokenizer.tokenizer_test(tok)

    # Get the vocab size. +1 is due to [PAD] token
    vocab_size = tok.vocab_size + 1

    # Shuffle the dataset using a seed
    ds = ds.shuffle(seed=2337)

    # Set the LOAD flag to True to load either latest model or a specified model
    # Set the LOAD flag to False to generate a default model
    LOAD = True

    if LOAD:
        # Initialize the file variable as None
        file = None

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
    num_batches = config.num_batches
    seq_len = config.seq_len

    # Train the model
    train(model, opt, ds, tok, loss, bs, num_batches, seq_len, model_params)

    # Save the model, optimizer, and model parameters
    utils.save_model(model, opt, model_params)


main()
