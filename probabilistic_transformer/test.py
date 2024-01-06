import torch.nn as nn
from tokenizers import Tokenizer

from transformer_libs import tokenizer
from transformer_libs import data_utils
from transformer_libs import utils
from transformer_libs import train
import os
import random
import pickle
from transformers import GPT2TokenizerFast

# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#
# Add special tokens
special_tokens_dict = {
    'pad_token': '[PAD]',
    'cls_token': '[CLS]',
    'sep_token': '[SEP]',
    'mask_token': '[MASK]',
    'unk_token': '[UNK]'
}



tok_file = "D:\data_sets\probabilistic_transformer\simple_wikipedia.json"
# Adding special tokens
toke = Tokenizer.from_file(tok_file)
#
num_added_tokens = toke.add_special_tokens(['[UNK]'])
print('Number of added tokens:', num_added_tokens)

# # Check if [UNK] is added successfully
# print('UNK token ID:', toke.convert_tokens_to_ids('[UNK]'))
# toke.add_tokens(['[UNK]'])
try:
    # This should not raise an error
    encoding = toke.encode("This is a test sentence dhfjkh___.")
    print("Encoded:", encoding)
    print("Token IDs:", encoding.ids)
    print("Decoded:", toke.decode(encoding.ids))
except Exception as e:
    print("Error during encoding:", str(e))

import pickle
file_path = "d:/data_sets/probabilistic_transformer/cleaned_joined_simple_wiki_sentences.pkl"
# Load your data
with open(file_path, 'rb') as f:
    data = pickle.load(f)


tokenized_articles = []
counter = 0
for article in data:
    counter += 1
    tokens = toke.encode(article)
    tokenized_articles.append(tokens.ids)
    if counter % 1000 == 0:
        print(counter)

file = "d:/data_sets/probabilistic_transformer/tokenized_simple_wiki_sentences.pkl"
with open(file, "wb") as f:
    pickle.dump(tokenized_articles, f)


file_path = "d:/data_sets/probabilistic_transformer/tokenized_simple_wiki_sentences.pkl"
# Load your data
with open(file_path, 'rb') as f:
    data = pickle.load(f)

import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        current_sample = []
        for article in self.data:
            for token in article:
                current_sample.append(token)
                if len(current_sample) == self.seq_len + 1:
                    samples.append(current_sample)
                    current_sample = []
            current_sample = []
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        src = torch.tensor(sample[:-1], dtype=torch.long)
        tgt = torch.tensor(sample[1:], dtype=torch.long)
        return src, tgt


seq_len = 256
dataset = TextDataset(data, seq_len)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print("loaded data")
for src, tgt in dataloader:
    print("Source:", src.shape)
    print("Target:", tgt.shape)
    print()

    # Loop through each sequence in the batch
    for i in range(src.shape[0]):
        # Convert token IDs to text, skipping padding tokens
        src_text = toke.decode(src[i].tolist(), skip_special_tokens=False)
        print('Source Text {}:'.format(i+1), src_text)

        # Convert target token IDs to text
        tgt_text = toke.decode(tgt[i].tolist(), skip_special_tokens=False)
        print('Target Text {}:'.format(i+1), tgt_text)


