import config
import torch
import torch.nn.functional as F
from transformer_libs import data_utils
from transformer_libs import utils

device = torch.device("cpu")

def prepare_train_config():
    model_params = config.model_params
    lr_params = config.lr_params
    train_params = config.train_params
    config_params = config.config_params
    return model_params, train_params, lr_params, config_params

def demo_model():
    torch.manual_seed(0)

    model_params, train_params, lr_params, config_params = prepare_train_config()

    # To load most recent model set LOAD flag to True
    LOAD = True
    # file = None To load a specific model uncomment this and set a file.
    model, opt, model_params, train_params, lr_params = utils.create_transformer_model(LOAD, config_params['model_directory'],
                                                                                       model_params, train_params, lr_params,
                                                                                       file=None)
    model.eval()

    utils.print_params(model_params)
    utils.print_params(train_params)
    utils.print_params(lr_params)
    utils.print_params(config_params)

    # Get the total number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {pytorch_total_params}')


    dl_params = {'max_seq_len': model_params['seq_len'], 'd_model': model_params['d_model'], 'vocab_size': model_params['vocab_size']}
    dl = data_utils.TransformerSeqDL(model_params['bs'], model_params['samples_done'], dl_params)
    src, target = dl()

    target = target.to(device)

    # run through model
    pred = model(src)

    pred = pred.permute(0, 2, 1)
    pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
    pred = pred[0,:]
    target = target[0,:]
    print(pred.shape)
    print(target.shape)
    print(pred)
    print(target)


demo_model()