import config
import torch
from transformer_libs import tokenizer
from transformer_libs import data_utils
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

    seq_len = 256
    bs = 2000
    samples_done = 0
    dl_params = {'L': seq_len, 'tok': tok, 'ds': ds}
    dl = data_utils.WikipediaDL_2(bs, samples_done, dl_params, tok, ds, pickle_files)

    src, target = next(iter(dl))

    #src = src.tolist()
    print(src)
    print(target)
    print(tok.decode(src[0]))
    print(tok.decode(target[0]))
    print(tok.decode(src[1]))
    print(tok.decode(target[1]))
if __name__ == "__main__":
    main()

from torch.nn import TransformerDecoderLayer

decoder_layer = TransformerDecoderLayer(d_model=layer_size, nhead=h, dim_feedforward=4 * layer_size, dropout=self.dropout_rate)
