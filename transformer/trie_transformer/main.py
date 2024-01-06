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

def show_sample_function_generator(tok):
    def show_sample(pred, target):
        probabilities = torch.softmax(pred, dim=-1)
        predicted_token_ids = torch.argmax(probabilities, dim=-1)

        predicted_token_ids_list = predicted_token_ids.tolist()
        target_sample_ids_list = target.tolist()

        # Only process the first element of the batch for now
        predicted_sequence = predicted_token_ids_list[0]
        target_sequence = target_sample_ids_list[0]

        # Decode sequences token by token
        for pred_token_id, target_token_id in zip(predicted_sequence, target_sequence):
            decoded_predicted = tok.decode([pred_token_id])
            decoded_target = tok.decode([target_token_id])
            print(f"Target: {decoded_target}\t Predicted: {decoded_predicted}")


    return show_sample

def prepare_train_config():
    model_params = config.model_params
    lr_params = config.lr_params
    lr_params['batch_num'] = 0
    train_params = config.train_params
    config_params = config.config_params
    return model_params, train_params, lr_params, config_params


def main():
    torch.manual_seed(0)

    # Load the dataset
    directory = config.ds_dir
    all_files = os.listdir(directory)

    # Filter for files with a .pkl extension
    pickle_files = [os.path.join(directory, file) for file in all_files if file.endswith('.pkl')]
    random.shuffle(pickle_files)
    ds = data_utils.load_ds(pickle_files[0])
    print(len(ds))
    random.shuffle(ds)
    # Load the tokenizer
    tok = tokenizer.load_tokenizer(config.tokenizer_class, config.tok_file, config.tok_special_tokens)
    vocab_size = tok.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    encoded, decoded = tok.test("Testing the tokenizer!")
    print(encoded, decoded)
    model_params, train_params, lr_params, config_params = prepare_train_config()
    model_params['vocab_size'] = vocab_size
    # To load most recent model set LOAD flag to True
    LOAD = False
    # file = None, To load a specific model uncomment this and set a file.
    model, opt, model_params, train_params, lr_params = utils.create_model(LOAD, config.model_directory,
                                                                           model_params, train_params, lr_params,
                                                                           file=None)



    train_params['batch_num'] = model_params['batch_num']

    # Get the total number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {pytorch_total_params}')

    # Set the loss function as cross-entropy loss
    def cross_entropy(pred, target):
        pred = pred.permute(0, 2, 1)
        loss_f = nn.CrossEntropyLoss()
        loss = loss_f(pred, target)
        return loss


    loss_func = cross_entropy
    lr_func = utils.constant_lr


    dl_params = {'L': model_params['seq_len'], 'tok': tok, 'ds': ds}
    dl = data_utils.WikipediaDL_2(model_params['bs'], model_params['samples_done'], dl_params, tok, ds, pickle_files)

    sample_function = show_sample_function_generator(tok)

    lr_params['constant_lr'] = 2.5e-4

    utils.print_params(model_params)
    utils.print_params(train_params)
    utils.print_params(lr_params)
    utils.print_params(config_params)

    trainer = train.Train(model, opt, model_params, train_params, lr_params, config_params, loss_func, dl,
                          lr_func, sample_function=sample_function)

    trainer.train(150_000)


main()
