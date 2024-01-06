import torch.nn.functional as F
import os
import config
import torch
import loss_functions
# from ProbabilisticTransformer import ProbabilisticTransformer as PT
from torch.utils.data import DataLoader

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch.nn as nn
import random
import pickle
import numpy as np
from transformer_libs import tokenizer
from transformer_libs import data_utils
from transformer_libs import utils
from transformer_libs import train
from transformer_libs.new_PT import ProbabilisticTransformer as PT


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        pos = self.encoding[:, :x.size(1)].detach().to(device)
        return x + pos


def probabilistic_transformer_loss_1(preds, target, beta=1.0, sigma_eps=1e-5, sigma_max_penalty=0.1,
                                     semantic_loss_weight=0.5):
    output_token, mu_out, sigma_out = preds
    target_token, target_mu = target

    # target_sigma needs to be computed.
    # Compute the token classification loss
    classification_loss = F.cross_entropy(output_token.view(-1, output_token.size(-1)), target_token.view(-1))

    # Compute the semantic similarity between predicted mu and target mu
    semantic_similarity = F.cosine_similarity(mu_out, target_mu, dim=-1).mean()

    # Reduce the penalty for classification loss based on semantic similarity
    adjusted_classification_loss = (1 - semantic_loss_weight * semantic_similarity) * classification_loss

    # Compute the cosine similarity loss for mu
    mu_loss = 1 - semantic_similarity

    abs_diff = torch.abs(mu_out - target_mu)

    # Compute the mean absolute deviation across the batch dimension
    approx_sigma = torch.mean(abs_diff, dim=0) + sigma_eps

    # Ensure approx_sigma has the same shape as sigma_out for loss calculation
    approx_sigma = approx_sigma.expand_as(sigma_out)

    # Compute the loss for sigma, encouraging it to be close to the approximated sigma
    sigma_loss = F.mse_loss(sigma_out, approx_sigma)

    # Add a penalty for very small values of sigma
    sigma_penalty = torch.where(sigma_out < sigma_eps, sigma_max_penalty * torch.ones_like(sigma_out),
                                torch.zeros_like(sigma_out))
    sigma_loss += sigma_penalty.mean()

    # Total loss is a weighted sum of the classification loss and the probabilistic losses
    total_loss = adjusted_classification_loss + beta * (mu_loss + sigma_loss)
    return total_loss, adjusted_classification_loss, mu_loss, sigma_loss, semantic_similarity


def probabilistic_transformer_loss_2(preds, target, alpha=1.0, beta=1.0, semantic_loss_weight=0.5):
    output_token, mu_out, sigma_out = preds
    target_token, target_mu = target

    # Compute the token classification loss
    classification_loss = F.cross_entropy(output_token.view(-1, output_token.size(-1)), target_token.view(-1))

    # Compute the semantic similarity between predicted mu and target mu
    semantic_similarity = F.cosine_similarity(mu_out, target_mu, dim=-1).mean()

    # Reduce the penalty for classification loss based on semantic similarity
    adjusted_classification_loss = (1 - semantic_loss_weight * semantic_similarity) * classification_loss

    # Compute the cosine similarity loss for mu
    mu_loss = 1 - semantic_similarity

    # Penalty for being confident but wrong
    # penalty = (mu_loss ** 2) / (sigma_out.mean() ** 2 + 1e-5)  # adding a small epsilon for numerical stability

    # Penalize large values of sigma
    sigma_penalty = alpha * sigma_out.mean()  # torch.exp(alpha * sigma_out.mean())

    # Total loss is a weighted sum of the classification loss, mu loss, sigma penalty, and penalty
    total_loss = adjusted_classification_loss + mu_loss + beta * (sigma_penalty)  # + penalty)

    return total_loss, adjusted_classification_loss, mu_loss, semantic_similarity


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_sample_function_generator(tok):
    def show_sample(pred, target):
        logits, sigmas = pred
        probabilities = torch.softmax(logits, dim=-1)
        predicted_token_ids = torch.argmax(probabilities, dim=-1)

        predicted_token_ids_list = predicted_token_ids.tolist()
        target_sample_ids_list = target.tolist()

        # Decode each sequence in the batch
        for i in range(1):
            predicted_sequence = predicted_token_ids_list[i]
            target_sequence = target_sample_ids_list[i]

            # Decode sequences
            decoded_predicted = tok.decode(predicted_sequence)
            decoded_target = tok.decode(target_sequence)

            print(f"Predicted Sequence {i+1}: {decoded_predicted}")
            print(f"Target Sequence {i+1}: {decoded_target}\n")
    return show_sample

def create_model(model_params, train_params, lr_params):
    print('Generating a new model to train.')
    model_params['id'] = "123"
    model = PT(**model_params)
    train_params['batch_num'] = 0
    train_params['samples_done'] = 0
    opt = torch.optim.Adam(model.parameters(), lr=lr_params['lr'])
    return model, opt, model_params, train_params, lr_params


def custom_loss():
    return None


def prepare_train_config(model_parameters_old=None, train_params_old=None, lr_params_old=None, config_parameters_old=None):
    model_params = config.model_params
    lr_params = config.lr_params
    train_params = config.train_params
    config_params = config.config_params

    # Update model parameters with old ones if not present
    if model_parameters_old:
        for k, v in model_parameters_old.items():
            if k not in model_params:
                model_params[k] = v

    # Update training parameters with old ones if not present
    if train_params_old:
        for k, v in train_params_old.items():
            if k not in train_params:
                train_params[k] = v

    # Update learning rate parameters with old ones if not present
    if lr_params_old:
        for k, v in lr_params_old.items():
            if k not in lr_params:
                lr_params[k] = v

    # Update config parameters with old ones if not present
    if config_parameters_old:
        for k, v in config_parameters_old.items():
            if k not in config_params:
                config_params[k] = v

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
    utils.print_params(model_params)
    utils.print_params(train_params)
    utils.print_params(lr_params)
    utils.print_params(config_params)

    # Get the total number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {pytorch_total_params}')

    train_output = None #new_PT.create_train_output(tok)

    # Set the loss function and parameters
    loss_func = loss_functions.surprise_loss
    lr_func = utils.get_lr
    lr_params['constant_lr'] = config.lr_params['constant_lr']
    lr_params['lr'] = config.lr_params['lr']

    dl_params = {'L': model_params['seq_len'], 'tok': tok, 'ds': ds}
    dl = data_utils.WikipediaDL_2(model_params['bs'], model_params['samples_done'], dl_params, tok, ds, pickle_files)

    sample_function = show_sample_function_generator(tok)

    utils.save_model(model, opt, config.model_directory, model_params, train_params, lr_params)

    trainer = train.Train(model, opt, model_params, train_params, lr_params, config_params, loss_func, dl,
                          lr_func, sample_function=sample_function)

    trainer.train(150_000)



if __name__ == "__main__":
    main()
