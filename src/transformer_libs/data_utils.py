import random
import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text_to_token(text, tokenizer):
    output = tokenizer.encode(text).ids
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


class BaseDataLoader:
    def __init__(self, bs, samples_done, params):
        self.bs = bs
        self.samples_done = samples_done
        for key, value in params.items():
            setattr(self, key, value)

    def update(self, params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __call__(self):
        src = None
        target = None
        return src, target


# wikipedia dl
class WikipediaDL(BaseDataLoader):
    def __init__(self, bs, samples_done, params):
        super().__init__(bs, samples_done, params)
        self.ds = None
        self.tok = None
        self.L = None
        for key, value in params.items():
            setattr(self, key, value)

    def __call__(self):
        low = (self.samples_done * self.bs) % len(self.ds)
        high = ((self.samples_done + 1) * self.bs) % len(self.ds)

        if low > high:
            sample = self.ds[low:] + self.ds[:high]
        else:
            sample = self.ds[low:high]

        sample = self.tok.get_tok()(
            sample,
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors="pt"
        ).input_ids

        if (self.samples_done + 1) % 1000 == 0:
            total_seq = self.samples_done * self.bs * self.L
            print(f"\n***** Epoch progress: {total_seq/len(self.ds)} ***** \n")

        combined = sample[:, :self.L]
        src = combined[:, :-1].to(device)  # bs x L
        target = combined[:, 1:].to(device)  # bs x L
        return src, target


class BooksDL(BaseDataLoader):
    def __init__(self, bs, samples_done, params):
        super().__init__(bs, samples_done, params)
        self.ds = None
        self.tok = None
        self.L = None
        for key, value in params.items():
            setattr(self, key, value)

    def __call__(self):
        low = (self.samples_done * self.bs) % len(self.ds)
        high = ((self.samples_done + 1) * self.bs) % len(self.ds)

        if low > high:
            sample = self.ds[low:] + self.ds[:high]
        else:
            sample = self.ds[low:high]

        sample = self.tok.get_tok()(
            sample,
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors="pt"
        ).input_ids

        if (self.samples_done + 1) % 1000 == 0:
            total_seq = self.samples_done * self.bs * self.L
            print(f"\n***** Epoch progress: {total_seq/len(self.ds)} ***** \n")

        combined = sample[:, :self.L]
        src = combined[:, :-1].to(device)  # bs x L
        target = combined[:, 1:].to(device)  # bs x L
        return src, target


class ChatBotDL(BaseDataLoader):
    def __init__(self, bs, samples_done, params):
        super().__init__(bs, samples_done, params)
        self.ds = None
        self.tok = None
        self.L = None
        for key, value in params.items():
            setattr(self, key, value)

    def __call__(self):
        low = (self.samples_done * self.bs) % len(self.ds)
        high = ((self.samples_done + 1) * self.bs) % len(self.ds)

        if low > high:
            sample = self.ds[low:] + self.ds[:high]
        else:
            sample = self.ds[low:high]

        sample = self.tok.get_tok()(
            sample,
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors="pt"
        ).input_ids
        combined = sample[:, :self.L]
        src = combined[:, :-1].to(device)  # bs x L
        target = combined[:, 1:].to(device)  # bs x L
        return src, target

# This is a data loader for the transformer
# It is specialized for the task of reversing a sequence of numbers
class TransformerSeqDL(BaseDataLoader):
    def __init__(self, bs, samples_done, params):
        super().__init__(bs, samples_done, params)
        self.max_seq_len = None
        self.vocab_size = None
        for key, value in params.items():
            setattr(self, key, value)

    def __call__(self):
        seq_len = random.randint(1, self.max_seq_len)
        vocab_size = self.vocab_size
        bs = self.bs
        r = torch.randint(1, vocab_size - 1, (bs, seq_len))

        enc_src, dec_src, target = r.clone(), r.clone(), r.clone()

        et = (vocab_size - 1) * torch.ones(bs, 1)
        st = torch.zeros(bs, 1)

        enc_src = torch.cat((enc_src, et), 1).long()
        dec_src = torch.cat((st, dec_src.flip(1)), 1).long()
        target = torch.cat((target.flip(1), et), 1).long()

        return (enc_src, dec_src), target


# DL for next word Shakespeare generation
class ShakespeareDL(BaseDataLoader):
    def __init__(self, bs, samples_done, params):
        super().__init__(bs, samples_done, params)
        self.ds = None
        self.L = None
        self.total_sequences = 0
        self.samples_done = samples_done
        for key, value in params.items():
            setattr(self, key, value)

    def __call__(self):
        low = (self.samples_done * (self.bs * self.L)) % len(self.ds)
        high = ((self.samples_done + 1) * (self.bs * self.L)) % len(self.ds)

        if low > high:
            sample = torch.cat([torch.Tensor(self.ds[low:]), torch.Tensor(self.ds[:high])], dim=-1)
            sample = sample.long()

        else:
            sample = self.ds[low:high]
            sample = torch.Tensor(sample).long()

        if (self.samples_done + 1) % 1000 == 0:
            total_seq = self.samples_done * self.bs * self.L
            print(f"\n***** Epoch progress: {total_seq/len(self.ds)} *****\n")

        self.total_sequences += self.bs * self.L
        sample = sample.view(self.bs, self.L)
        combined = sample[:, :self.L]
        src = combined[:, :-1].to(device)  # bs x L
        target = combined[:, 1:].to(device)  # bs x L
        return src, target


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
    combined = torch.gather(sample, 1, idx).to(device)
    src = combined[:, :-1].to(device)  # bs x L
    target = combined[:, 1:].to(device)  # bs x L
    return src, target

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
    bad_chars = {ord('/'): None, ord('\n'): " "}
    s = s.translate(bad_chars)
    return s


import re


def validate_line(line):
    x = re.search(r"[^\w\d\s,\.:\'\?!;-]|[_]", line)
    if x:
        return False
    else:
        return True


def process_string(b, max_seq_len):
    b = ''.join(b)
    b = b.split('.')
    b = [x + '.' for x in b]
    seq = ''
    result = []
    total_words = 0
    for sentence in b:
        seq = seq.strip()
        sentence = remove_bad_chars(sentence)
        if not validate_line(sentence):
            continue
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
    count = 0
    while len(ds) < min_data_len:
        # get random directory
        d = directories[np.random.randint(2, len(directories))]
        path_d = book_path + d
        files = get_files_in_dir(path_d)
        file = files[np.random.randint(0, len(files))]
        full_path = path_d + '\\' + file

        b = load_book(full_path)
        b = process_string(b, 256)
        ds.extend(b)
        count += 1
        if (count + 1) % 100 == 0:
            print("percent done: ", len(ds) / db_size)
    random.shuffle(ds)
    return ds


import pickle


def save_ds(ds, file):
    open_file = open(file, "wb")
    pickle.dump(ds, open_file)


def load_ds(file):
    print("Loading pickle file")
    with open(file, 'rb') as f:
        return pickle.load(f)


def create_book_lists(ds_size):
    v_size = 100_000
    test_size = 100_000

    path = "C:\\Users\\semis\\IdeaProjects\\DL\\transformer\\transformer-books\\data\\"
    s_file = path + "book_list2.pkl"
    ds = create_book_ds(ds_size)
    ds = ds[:ds_size]
    #
    save_ds(ds, s_file)
    #
    file = path + "book_list2.pkl"
    t_file = path + 'book_train2.pkl'
    v_file = path + 'book_valid2.pkl'
    test_file = path + 'book_test2.pkl'
    ds = load_ds(file)
    print(len(ds))
    train = ds[:-v_size - test_size]
    valid = ds[ds_size - (v_size + test_size): ds_size - v_size]
    test = ds[ds_size - v_size:]
    save_ds(train, t_file)
    save_ds(valid, v_file)
    save_ds(test, test_file)
    print(len(train))
    print(len(test))
    print(len(valid))


import pandas as pd


def save_chat_ds(ds, file):
    open_file = open(file, "wb")
    pickle.dump(ds, open_file)


def create_chat_data(tok):
    df = pd.read_csv("./data/topical_chat.csv")
    pad = 32767
    eot = 0
    conv_ds = []
    overflow = []
    last_id = df['conversation_id'].tolist()[-1]
    for m_id in range(1, last_id):
        messages = df.loc[df['conversation_id'] == m_id]['message']
        conv = []
        for m in messages:
            tok_m = tok.encode(m)
            tok_m.append(eot)
            if len(tok_m) > 256:
                tok_m = tok_m[:256]
            if len(conv) + len(tok_m) <= 256:
                conv.extend(tok_m)
            else:
                overflow = tok_m
                while (len(conv)) < 257:
                    conv.append(pad)
                conv_ds.append(conv)
                conv = overflow[:]
                overflow = []
            if len(conv) == 256:
                conv.append(pad)
                conv_ds.append(conv)
                conv = overflow[:]
                overflow = []
        if len(conv) > 0:
            while (len(conv)) < 257:
                conv.append(pad)
            conv_ds.append(conv)

    return conv_ds


def create_wikipedia_ds(wiki_ds):
    ds = []
    count = 0
    for article in range(0, len(wiki_ds['train'])):
        article = wiki_ds['train'][article]['text']
        b = process_string(article, 256)
        ds.extend(b)
        count += 1
        if (count + 1) % 100 == 0:
            print("percent done: ", len(ds) / len(wiki_ds['train']))

    path = "C:\\Users\\semis\\IdeaProjects\\DL\\transformer\\shared_data\\"
    s_file = path + "wiki_list.pkl"

    save_ds(ds, s_file)
    #
    file = path + "wiki_train.pkl"
    t_file = path + 'wiki_train.pkl'
    v_file = path + 'wiki_valid.pkl'
    test_file = path + 'wiki_test.pkl'
    ds = load_ds(file)
    print(len(ds))
    v_size = 10_000
    test_size = 10_000
    ds_size = len(ds)
    train = ds[:-v_size - test_size]
    valid = ds[ds_size - (v_size + test_size): ds_size - v_size]
    test = ds[ds_size - v_size:]
    save_ds(train, t_file)
    save_ds(valid, v_file)
    save_ds(test, test_file)
    print(len(train))
    print(len(test))
    print(len(valid))


def load_wikipedia():
    from datasets import load_dataset
    ds = load_dataset("wikipedia", "20220301.en")
    return ds


def get_sequence_batch(bs, seq_len, d_model):
    seq_len = random.randint(1, seq_len)

    r = torch.randint(1, d_model - 1, (bs, seq_len))

    enc_src, dec_src, target = r.clone(), r.clone(), r.clone()

    et = (d_model - 1) * torch.ones(bs, 1)
    st = torch.zeros(bs, 1)

    enc_src = torch.cat((enc_src, et), 1)
    dec_src = torch.cat((st, dec_src.flip(1)), 1)
    target = torch.cat((target.flip(1), et), 1).long()

    enc_src = F.one_hot(enc_src.to(torch.int64), num_classes=d_model).float()
    dec_src = F.one_hot(dec_src.to(torch.int64), num_classes=d_model).float()

    enc_src = enc_src.to(device)
    dec_src = dec_src.to(device)
    target = target.to(device)

    return enc_src, dec_src, target


def read_shakespeare_data(shakespeare_path):
    df = pd.read_csv(shakespeare_path)

    lines = df['PlayerLine'].tolist()
    lines = [line + ' ' for line in lines]

    # words = [x.split() for x in lines]
    #
    # result = []
    #
    # for word in words:
    #     if len(word) > 2:
    #         result.extend(word)
    #
    # final = []
    # for w in result:
    #     w = w.lower()
    #     final.extend(re.findall(r"\w+|[^\w\s]", w, re.UNICODE))

    return lines

def create_shakespeare_ds(shakespeare_path, tok, file):
    df = pd.read_csv(shakespeare_path)
    lines = df['PlayerLine'].tolist()
    result = []
    for line in lines:
        result.extend(tok.encode(line))
    save_ds(result, file)
    return result




if __name__ == '__main__':
    read_shakespeare_data("../../transformer/transformer-shakespeare/data/shakespeare/shakespeare.csv")
