import torch
import pandas as pd
import re
def read_shakespeare_data(shakespeare_path):

    df = pd.read_csv(shakespeare_path)

    lines = df['PlayerLine'].tolist()

    words = [x.split() for x in lines]

    result = []

    for word in words:
        if len(word) > 2:
            result.extend(word)

    final = []
    for w in result:
        w = w.lower()
        final.extend(re.findall(r"\w+|[^\w\s]", w, re.UNICODE))

    return final


class DataLoader:
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.current_index = 0

    def get_slice(self):
        src = self.data[self.current_index:self.current_index+self.seq_len]
        target = self.data[self.current_index+1:self.current_index+self.seq_len+1]
        self.current_index += self.seq_len
        return src, target

    def get_data(self):
        if self.current_index > len(self.data) - self.seq_len:
            self.current_index = 0
        return self.get_slice()

    def get_batch(self, bs):

        src = []
        target = []
        for _ in range(bs):

            s, t = self.get_data()
            src.append(s.detach().unsqueeze(0).clone())
            target.append(t.detach().unsqueeze(0))
        src = torch.cat(src, dim=0)
        target = torch.cat(target, dim=0)
        return src, target


def text_to_token(text, tokenizer):
    output = tokenizer.encode(text)
    return output.ids

def tokens_to_tensor(tokens):
    return torch.LongTensor(tokens.ids)

def tensor_to_token_ids(t):
    t = t.squeeze()
    t = t.tolist()
    return t

def token_id_to_word(t, tokenizer):
    return tokenizer.id_to_token(t)

def tokens_to_words(t, tokenizer):
    result = ""
    for id in t:
        result += " " + token_id_to_word(id, tokenizer)
    return result

def text_to_model_input(text, tokenizer):
    tokens = text_to_token(text, tokenizer)
    return torch.LongTensor(tokens)