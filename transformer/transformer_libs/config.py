# each program using files from transformer_lib needs to have theor own config file
# Model parameters
# size of the model
d_model = -1
# number of heads in the model
h = 1

# dictionary containing model parameters
model_params = {'num_blocks': 0,
                'd_model': d_model,
                'd_middle': 4 * d_model,
                'dropout': 0.1,
                'h': 1,
                'd_q': d_model // h,
                'd_k': d_model // h,
                'd_v': d_model // h,
                }

# maximum sequence length
seq_len = 0

# directory for storing the model
model_directory = '.\\models\\'

# batch size for training
batch_size = 0

# zero gradients after every accumulate_size batches
accumulate_size = 1

effective_bs = batch_size * accumulate_size

# total number of batches for training
num_batches = 0

# path for dataset
ds_path = "wikipedia"
ds_file = "20220301.en"

# tokenizer file path
tok_file = "./data/tokenizer-32768.json"

# directory for storing log files
log_directory = "./logs/"
log_file = "log.txt"

# frequency at which model is saved during training
save_every = -1

# frequency at which output is generated during training
output_every = -1

# frequency at which learning rate is updated during training
lr_step = -1
batch_scale_factor = -1

print("Warning you are importing the default config file.  This config should be overwritten by your own config file.")