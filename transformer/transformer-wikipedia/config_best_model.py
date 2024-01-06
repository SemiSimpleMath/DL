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
accumulate_size = 50
train_params = {
    'accumulate_size': accumulate_size,
    'update_lr_every': accumulate_size,
    'output_every': 10 * accumulate_size,
    'save_every': 50 * accumulate_size,
    'eval_every': 20000 * accumulate_size,
}


lr_params = {
    'update_lr_every': accumulate_size,
    'batch_scale_factor': 1,
    'constant_lr': 2.5e-4,
    'lr': 2.5e-4,
}

config_params = {
    'model_directory': './models/',
    'log_directory': './logs/'
}
# config.py
from transformer_libs.tokenizer import WikiTok
tokenizer_class = WikiTok
tok_special_tokens = [('pad_token', '[PAD]')]
tok_file = "../shared_data/tokenizers/tokenizer-32768.json"

wiki_ds_file = "C:\\Users\\semis\\IdeaProjects\\DL\\transformer\\shared_data\\" + 'wiki_train.pkl'

#wiki_ds_file = "d:/data_sets/probabilistic_transformer/tokenized_simple_wiki_sentences.pkl"


#tok_file = "D:/data_sets/probabilistic_transformer/simple_wikipedia.json"

# directory for train logs
log_directory = './logs/'

# save and load models from
model_directory = './models/'

