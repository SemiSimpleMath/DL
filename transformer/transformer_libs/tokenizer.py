from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import GPT2TokenizerFast, AutoTokenizer

import config

from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)


def tokenizer_train_wiki_103():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    tokenizer.pre_tokenizer = Whitespace()

    files = [f"./data/wikitext-103-raw-v1/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
    tokenizer.train(files, trainer)

    return tokenizer


def tokenizer_save(tok):
    tok.save("data/tokenizer-32768-2.json")


def tokenizer_load(path):
    tokenizer = Tokenizer.from_file(path)
    return tokenizer


def load_wikipedia():
    from datasets import load_dataset
    ds = load_dataset("wikipedia", "20220301.en")
    return ds


def batch_iterator(ds, bs=1_000):
    i = 0
    while i < len(ds['train']) // bs:
        if i % 10 == 0:
            print(i)
        i += 1
        yield ds['train'][i * bs: i * bs + bs]['text']


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


def train_tokenizer():
    ds = load_wikipedia()
    tok = Tokenizer(models.BPE())

    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["<|endoftext|>"])

    trainer = trainers.BpeTrainer(vocab_size=32767, special_tokens=["<|endoftext|>"])

    tok.train_from_iterator(batch_iterator(ds), trainer=trainer)

    tokenizer_save(tok)

    encoding = tok.encode("Let's test this tokenizer.")
    print(encoding.tokens)

    tok.decoder = decoders.ByteLevel()

    from transformers import GPT2TokenizerFast

    wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tok)

    return wrapped_tokenizer


def load_tokenizer():
    toke = Tokenizer.from_file(config.tok_file)
    toke.decoder = decoders.ByteLevel()
    tok = GPT2TokenizerFast(tokenizer_object=toke)
    tok.add_special_tokens({'pad_token': '[PAD]'})
    return tok


def tokenizer_test(tok):
    encoding = tok.encode("Let's test this tokenizer.")
    print(encoding)
    print(tok.decode(encoding))


if __name__ == '__main__':
    tok = train_tokenizer()
    tok.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer_test(tok)
