import random

import torch
import numpy as np
import sys

# setting path
sys.path.append('../transformer_libs')
import utils


def text_to_token(text, tokenizer):
    output = tokenizer.encode(text)
    return output


def tokens_to_tensor(tokens):
    return torch.LongTensor(tokens.ids)


def tensor_to_token_ids(t):
    t = t.squeeze()
    t = t.tolist()
    return t


def token_id_to_word(t, tokenizer):
    return tokenizer.id_to_token(t)


def tokens_to_words(t, tok):
    return tok.decode(t)


def text_to_model_input(text, tokenizer):
    tokens = text_to_token(text, tokenizer)
    return torch.LongTensor(tokens)


def get_wiki_batch(ds, tok, bs, batches_done, L):
    sample = tok(
        ds['train'][batches_done * bs: (batches_done + 1) * bs]['text'],
        padding=True,
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_tensors="pt"
    ).input_ids
    firstpad = sample.argmax(dim=-1)
    starts = (torch.rand(bs) * torch.clamp(firstpad - L + 1, min=0).to(torch.float)).to(torch.int64)
    if firstpad.max() < L:
        print(f"firstpad.max() < L! {firstpad.max()} {L}")
        print(sample.shape)
        print(sample)
    idx = starts.unsqueeze(-1) + torch.arange(min(L, firstpad.max()))
    combined = torch.gather(sample, 1, idx).to(utils.device)

    return combined

def get_batch(ds, tok, bs, samples_done, L):
    low = samples_done * bs % len(ds)
    high = (samples_done + 1) * bs % len(ds)

    if low >= high:
        low = 0
        high = bs

    sample = tok(
        ds[low: high],
        padding=True,
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_tensors="pt"
    ).input_ids
    firstpad = sample.argmax(dim=-1)
    starts = (torch.rand(bs) * torch.clamp(firstpad - L + 1, min=0).to(torch.float)).to(torch.int64)
    if firstpad.max() < L:
        print(f"firstpad.max() < L! {firstpad.max()} {L}")
        print(sample.shape)
        print(sample)
    idx = starts.unsqueeze(-1) + torch.arange(min(L, firstpad.max()))
    combined = torch.gather(sample, 1, idx).to(utils.device)

    return combined

def load_book(file):
    book = []
    with open(file, encoding="utf8") as f:
        for line in f:
            if len(line) > 40:
                book.append(line)
    return book


from os import listdir
from os.path import isfile, join


def get_files_in_dir(d):
    files = [f for f in listdir(d) if isfile(join(d, f))]
    return files


def get_directories_in_path(p):
    return listdir(p)


def remove_bad_chars(s):
    bad_chars = {ord('_'): None, ord('/'): None, ord('\n'): " "}
    s = s.translate(bad_chars)
    return s


def process_book(b, max_seq_len):
    b = ''.join(b)
    b = b.split('.')
    sequence_len = 0
    seq = ''
    result = []
    total_words = 0
    for sentence in b:
        seq = seq.strip()
        sentence = remove_bad_chars(sentence)
        sentence = sentence.strip()
        words_sentence = len(sentence.split())
        if words_sentence < 5:
            continue
        seq = seq + ' ' + sentence
        total_words += words_sentence
        if total_words > max_seq_len:
            result.append(seq)
            seq = ''
            total_words = 0
    return result


def create_book_ds(db_size=100_000):
    book_path = "D:\\data sets\\book3\\books3.tar\\books3\\books3\\the-eye.eu\\public\\Books\\Bibliotik\\"

    ds = []
    directories = get_directories_in_path(book_path)
    min_data_len = db_size
    while len(ds) < min_data_len:
        # get random directory
        d = directories[np.random.randint(2, len(directories))]
        path_d = book_path + d
        files = get_files_in_dir(path_d)
        file = files[np.random.randint(0, len(files))]
        full_path = path_d + '\\' + file

        b = load_book(full_path)
        b = process_book(b, 256)
        ds.extend(b)
    random.shuffle(ds)
    return ds
