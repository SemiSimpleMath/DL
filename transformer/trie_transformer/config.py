# Model parameters
# size of the model
d_model = 40
# number of heads in the model
h = 12

# dictionary containing model parameters
model_params = {
                'num_embd_layers': 1,
                'base_embd_size': 768,
                'num_blocks_embed': 4,
                'embed_heads': h,
                'dropout': 0.1,
                'tail_heads': h,
                'num_tail_blocks': 8,
                'weight_decay': 0.1,
                'betas':  (0.9, 0.95),
                'seq_len': 256,
                'bs': 16,
                'weight_tying': False,
                'model_class_name': 'trie_transformer'
                }

# zero gradients after every accumulate_size batches
accumulate_size = 10
train_params = {
    'accumulate_size': accumulate_size,
    'update_lr_every': accumulate_size,
    'output_every': 50 * accumulate_size,
    'save_every': 1000 * accumulate_size,
    'eval_every': 200000 * accumulate_size,
}


lr_params = {
    'update_lr_every': accumulate_size,
    'batch_scale_factor': 1,
    'constant_lr': 2.5e-4,
    'lr': 2.5e-4,
    'warmup_steps': 4000 // accumulate_size,
    'betas':  (0.9, 0.95),
}

config_params = {
    'model_directory': 'f:/models/trie_models',
    'log_directory': './logs/'
}
# config.py
from transformer_libs.tokenizer import WikiTok
tokenizer_class = WikiTok
tok_special_tokens = [('pad_token', '[PAD]')]
tok_file = "../shared_data/tokenizers/tokenizer-32768.json"

ds_dir = "D:/data_sets/probabilistic_transformer/tokenized/wikipedia"

# directory for train logs
log_directory = './logs/'

# save and load models from
model_directory = 'f:/models/trie_models'

