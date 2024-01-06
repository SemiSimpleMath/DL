
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import GPT2TokenizerFast


def load_wikipedia():
    from datasets import load_dataset
    ds = load_dataset("wikipedia", "20220301.en")
    return ds


def batch_iterator(ds, bs=1_000):
    i = 0
    while i * bs < len(ds):
        if i % 10 == 0 and i > 0:
            print(f'Processing batch {i}')
        batch = [ds[j]['text'] for j in range(i * bs, min((i + 1) * bs, len(ds)))]
        yield batch
        i += 1


def list_batch_iterator(ds, bs=1_000):
    i = 0
    while i * bs < len(ds):
        if i % 10 == 0 and i > 0:
            print(f'processing batcn {i}')
        start_index = i * bs
        end_index = min((i + 1) * bs, len(ds))
        batch = ds[start_index:end_index]
        yield batch
        i += 1


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

    def test(self,s):
        encoded = self.encode(s)
        decoded = self.decode(encoded)
        return encoded, decoded

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


class simple_WikiTok():
    def __init__(self, tok=None):
        if tok is not None:
            self.tok = tok

    def tokenizer_test(self, s="Let's test this tokenizer!"):
        encoding = self.tok.encode(s)
        print("Encoding:", encoding.ids)

    def load(self, file):
        tok_file = "D:\data_sets\probabilistic_transformer\simple_wikipedia.json"
        # Adding special tokens
        self.tok = Tokenizer.from_file(tok_file)

        return self.tok

    def get_tok(self):
        return self.tok

    def save(self, file):
        self.tok.save(file)
        print(f'Saved tokenizer: {file}')

    def get_vocab_size(self):
        # Get the vocab size. +1 is due to [PAD] token which we force to be the last token
        return len(self.tok.get_vocab()) + 1

    def train(self, ds, vocab_size, save_file):
        tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))

        # Optional: Use a whitespace pre-tokenizer
        tok.pre_tokenizer = pre_tokenizers.Whitespace()

        # Define the trainer
        trainer = trainers.WordLevelTrainer(vocab_size=vocab_size,
                                            special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"])

        tok.train_from_iterator(list_batch_iterator(ds), trainer=trainer)
        # Add special tokens to base tokenizer
        special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
        tok.add_tokens(special_tokens)
        # Save the tokenizer
        self.tok = tok
        tok.save(save_file)

        return self.tok

    def text_to_tensor(self, text):
        tokens = self.tok.encode(text)
        tensor = torch.tensor(tokens.ids, dtype=torch.long)
        return tensor

    def decode(self, data):
        return self.tok.decode([data])


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


class ByteLevelTok(BaseTok):
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

        tok.train_from_iterator(list_batch_iterator(ds), trainer=trainer)

        self.tok = tok

        tok.decoder = decoders.ByteLevel()

        from transformers import GPT2TokenizerFast

        wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tok)

        self.save(save_file)

        return self.tok, wrapped_tokenizer


if __name__ == '__main__':
    # initialize the tokenizer class instance
    tok = ByteLevelTok()
    # load the file to be tokenized
    tok_file = "d:\data_sets\probabilistic_transformer\simple_wikipedia_byte_level.json"
    import pickle

    file_path = "d:/data_sets/probabilistic_transformer/cleaned_joined_simple_wiki_sentences.pkl"
    # Load your data
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    tok = tok.train(data, 32767, tok_file)

# In a separate file, e.g., tokenizer_utils.py

def load_tokenizer(tokenizer_class, tok_file, additional_special_tokens):
    """
    Load the tokenizer with specified parameters.

    :param tokenizer_class: The class of the tokenizer to be instantiated.
    :param tok_file: File path to load the tokenizer.
    :param additional_special_tokens: List of additional special tokens to add.
    :return: An instance of the tokenizer.
    """
    tok = tokenizer_class()
    tok.load(tok_file)
    num_tokens_added = 0
    for token in additional_special_tokens:
        name, symbol = token
        tok.tok.add_special_tokens({name:symbol})
        num_tokens_added += 1

    print(f"Number of added special tokens: {num_tokens_added}")

    vocab_size = tok.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    return tok
