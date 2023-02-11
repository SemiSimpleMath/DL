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
                }

# maximum sequence length
seq_len = 256

# directory for storing the model
model_directory = '.\\models\\'

# batch size for training
batch_size = 16

# total number of batches for training
num_batches = 500000

# path for dataset
ds_path = "wikipedia"
ds_file = "20220301.en"

# tokenizer file path
tok_file = "../shared_data/tokenizers/tokenizer-32768.json"

# directory for storing log files
log_directory = "./logs/"
log_file = "log.txt"


# zero gradients after every accumulate_size batches
accumulate_size = 4

# frequency at which learning rate is updated during training
lr_step = accumulate_size
batch_scale_factor = 1

update_lr_every = accumulate_size
# frequency at which model is saved during training
save_every = 100 * accumulate_size

# frequency at which output is generated during training
output_every = 20 * accumulate_size

# frequency at which learning rate is updated during training

eval_every = 500 * accumulate_size
