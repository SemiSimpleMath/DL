from transformer_libs import new_PT

# model
model_class = new_PT.ProbabilisticTransformer
# Model parameters
# size of the model
d_model = 300
# number of heads in the model
num_heads = 10

assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
# dictionary containing model parameters
model_params = {'num_blocks': 8,
                'd_model': d_model,
                'd_middle': 4 * d_model,
                'dropout': 0.1,
                'num_heads': num_heads,
                'd_q': d_model // num_heads,
                'd_k': d_model // num_heads,
                'd_v': d_model // num_heads,
                'weight_decay': 0.1,
                'betas':  (0.9, 0.95),
                'seq_len': 128,
                'bs': 8,
                'weight_tying': False,
                'model_class_name': 'probabilistic_transformer'
                }

# zero gradients after every accumulate_size batches
accumulate_size = 10
train_params = {
    'accumulate_size': accumulate_size,
    'update_lr_every': accumulate_size,
    'output_every': 50 * accumulate_size,
    'save_every': 1000 * accumulate_size,
    'eval_every': 20000 * accumulate_size,
}


lr_params = {
    'update_lr_every': accumulate_size,
    'batch_scale_factor': 1,
    'constant_lr': 2.5e-4,
    'lr': 2.5e-5,
    'batch_num':0,
    'd_model': d_model,
    'step_size': 1000,
    'warmup_steps': 4000//accumulate_size

}
# directory for train logs
log_directory = 'd:/probabilistic_transformer/logs'

# save and load models from
model_directory = 'd:/probabilistic_transformer/models4/'
config_params = {
    'model_directory': model_directory,
    'log_directory': log_directory
}


################## TOKENIZER ########################
from transformer_libs.tokenizer import WikiTok
tokenizer_class = WikiTok
tok_special_tokens = [('pad_token', '[PAD]')]
tok_file = "../transformer/shared_data/tokenizers/tokenizer-32768.json"

# DATA FILE
ds_dir = "D:/data_sets/probabilistic_transformer/tokenized/wikipedia"


