from transformers import GPT2TokenizerFast
from transformer_libs import data_utils

from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)

import torch

def load_wikipedia():
    from datasets import load_dataset
    ds = load_dataset("wikipedia", "20220301.en")
    return ds

def batch_iterator(ds, bs=1_000):
    i = 0
    while i < len(ds) // bs:
        if i % 10 == 0:
            print(i)
        i += 1
        yield ds[i * bs: i * bs + bs]


def large_vocab_train(ds, book_ds, bs=1_000):
    i = 0
    while i < len(ds['train']) // bs:
        if i % 10 == 0:
            print(i)
        i += 1
        yield ds['train'][i * bs: i * bs + bs]['text']

    j = 0
    while j < len(book_ds) // bs:
        if j % 10 == 0:
            print(j)
        j += 1
        yield book_ds[i * bs: i * bs + bs]


class BaseTok:
    def __init__(self, tok=None):
        self.tok = tok

    def tokenizer_test(self, s="Let's test this tokenizer!"):
        encoding = self.tok.encode(s)
        print(encoding)
        print(self.tok.decode(encoding))

    def load(self, file):
        self.tok = Tokenizer.from_file(file)
        return self.tok

    def save(self, file):
        self.tok.save(file)

    def encode(self, data):
        return self.tok.encode(data)

    def decode(self, data):
        return self.tok.decode(data)

    def text_to_tensor(self, text):
        tokens = self.tok.encode(text)
        return torch.LongTensor(tokens)
    def get_tok(self):
        return self.tok


    def train(self, ds, vocab_size, save_file):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])  # vocab_size = 32767
        tok.train_from_iterator(batch_iterator(ds), trainer=trainer)

        self.tok = tok

        tok.decoder = decoders.ByteLevel()
        self.save(save_file)

        return tok


class WikiTok(BaseTok):
    def __init__(self, tok=None):
        super().__init__(tok)
        if tok is not None:
            self.tok = tok

    def load(self, file):
        toke = Tokenizer.from_file(file)
        toke.decoder = decoders.ByteLevel()
        tok = GPT2TokenizerFast(tokenizer_object=toke)
        tok.add_special_tokens({'pad_token': '[PAD]'})
        self.tok = tok
        return tok

    def save(self, file):
        self.tok.save(file)
        print(f'Saved tokenizer: {file}')

    def get_vocab_size(self):
        # Get the vocab size. +1 is due to [PAD] token which we force to be the last token
        return self.tok.vocab_size + 1

    def train(self, ds, vocab_size, save_file):
        tok = Tokenizer(models.BPE())

        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(vocab_size=32767, special_tokens=["<|endoftext|>"])

        tok.train_from_iterator(batch_iterator(ds), trainer=trainer)

        self.tok = tok

        tok.decoder = decoders.ByteLevel()

        from transformers import GPT2TokenizerFast

        wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tok)

        self.save(save_file)

        return self.tok, wrapped_tokenizer

class BooksTok(BaseTok):
    def __init__(self, tok=None):
        super().__init__(tok)
        if tok is not None:
            self.tok = tok

    def load(self, file):
        toke = Tokenizer.from_file(file)
        toke.decoder = decoders.ByteLevel()
        tok = GPT2TokenizerFast(tokenizer_object=toke)
        tok.add_special_tokens({'pad_token': '[PAD]'})
        self.tok = tok
        return tok

    def save(self, file):
        self.tok.save(file)
        print(f'Saved tokenizer: {file}')

    def get_vocab_size(self):
        # Get the vocab size. +1 is due to [PAD] token which we force to be the last token
        return self.tok.vocab_size + 1

    def train(self, ds, vocab_size, save_file):
        tok = Tokenizer(models.BPE())

        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(vocab_size=32767, special_tokens=["<|endoftext|>"])

        tok.train_from_iterator(batch_iterator(ds), trainer=trainer)

        self.tok = tok

        tok.decoder = decoders.ByteLevel()

        from transformers import GPT2TokenizerFast

        wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tok)

        self.save(save_file)

        return self.tok, wrapped_tokenizer


class WikiAndBookTok(BaseTok):
    def __init__(self, tok=None):
        super().__init__(tok)
        if tok is not None:
            self.tok = tok

    def load(self, file):
        toke = Tokenizer.from_file(file)
        toke.decoder = decoders.ByteLevel()
        tok = GPT2TokenizerFast(tokenizer_object=toke)
        tok.add_special_tokens({'pad_token': '[PAD]'})
        self.tok = tok
        return tok


class ShakespeareTok(BaseTok):
    def __init__(self, tok=None):
        super().__init__(tok)
        if tok is not None:
            self.tok = tok

    def load(self, file):
        toke = Tokenizer.from_file(file)
        toke.decoder = decoders.ByteLevel()
        self.tok = GPT2TokenizerFast(tokenizer_object=toke)
        self.tok.add_special_tokens({'pad_token': '[PAD]'})
        return self.tok

    def get_vocab_size(self):
        # Get the vocab size. +1 is due to [PAD] token which we force to be the last token
        return self.tok.vocab_size + 1


if __name__ == '__main__':
    shakespeare_tok = ShakespeareTok()

    shakespeare_path = '../../transformer/transformer-shakespeare/data/shakespeare/shakespeare.csv'
    tok_file = "../../transformer/transformer-shakespeare/data/tokenizer/shakespeare_tok.json"
    s_data = data_utils.read_shakespeare_data(shakespeare_path)
    shakespeare_tok.train(s_data, 20_000, tok_file)
