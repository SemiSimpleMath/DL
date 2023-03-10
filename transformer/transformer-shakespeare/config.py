# Model parameters
# size of the model
d_model = 768
# number of heads in the model
h = 12

# dictionary containing model parameters
model_params = {'num_blocks': 12,
                'd_model': d_model,
                'd_middle': 4 * d_model,
                'dropout': 0.1,
                'h': 12,
                'd_q': d_model // h,
                'd_k': d_model // h,
                'd_v': d_model // h,
                'weight_decay': 0.1,
                'betas':  (0.9, 0.95),
                'seq_len': 256,
                'bs': 8,
                'weight_tying': True,
                }

# zero gradients after every accumulate_size batches
accumulate_size = 10
train_params = {
    'accumulate_size': accumulate_size,
    'update_lr_every': accumulate_size,
    'output_every': 20 * accumulate_size,
    'save_every': 100 * accumulate_size,
    'eval_every': 2000 * accumulate_size,
}


lr_params = {
    'update_lr_every': accumulate_size,
    'batch_scale_factor': 1,
    'constant_lr': 2.5e-4,
    'lr': 1.5e-4,
}

config_params = {
    'model_directory': './models/',
    'log_directory': './logs/'
}

# tokenizer file path
tok_file = "./data/tokenizer/shakespeare_tok.json"
shakespeare_data_file = "./data/shakespeare/shakespeare.csv"
ds = "./data/shakespeare/shakespeare_ds.pkl"
# directory for train logs
log_directory = './logs/'

# save and load models from
model_directory = './models/'
