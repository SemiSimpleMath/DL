# Model parameters
# size of the model
d_model = 256
# number of heads in the model
h = 8
num_blocks = 8
# dictionary containing model parameters
model_params = {'num_blocks': num_blocks,
                'd_model': d_model,
                'd_middle': 4 * d_model,
                'dropout': 0.0,
                'h': h,
                'd_q': d_model // h,
                'd_k': d_model // h,
                'd_v': d_model // h,
                'weight_decay': 0.0,
                'betas':  (0.9, 0.95),
                'seq_len': 100,
                'bs': 16,
                'weight_tying': True,
                'vocab_size': 22
                }

# zero gradients after every accumulate_size batches
accumulate_size = 1
train_params = {
    'accumulate_size': accumulate_size,
    'update_lr_every': accumulate_size,
    'output_every': 100 * accumulate_size,
    'save_every': 2000 * accumulate_size,
    'eval_every': 20000 * accumulate_size,
}


lr_params = {
    'update_lr_every': accumulate_size,
    'batch_scale_factor': 1,
    'constant_lr': 1.5e-4,
    'lr': 2.5e-4,
}

config_params = {
    'model_directory': './models/',
    'log_directory': './logs/'
}

wiki_ds_file = "C:\\Users\\semis\\IdeaProjects\\DL\\transformer\\shared_data\\" + 'wiki_train.pkl'

# tokenizer file path
tok_file = "../shared_data/tokenizers/tokenizer-32768.json"

log_directory = './logs/'
model_directory = './models/'

