import torch
import random
import utils


class DataLoader:
    def __init__(self, data, tok):
        self.ds = data
        self.tok = tok

    def get_batch(self, bs, min_seq_len):
        src_batch = []
        target_batch = []

        while len(src_batch) < bs:
            r = random.randint(0, 6458670 - 1)
            article = self.ds['train'][r]['text']
            article = article.split()
            art_len = len(article)
            if art_len < min_seq_len + 2:
                continue
            start = random.randint(0, art_len - min_seq_len - 1)
            source = article[start: start + min_seq_len + 2]
            source = self.tok.encode(' '.join(source))

            src = torch.tensor(source[0:min_seq_len]).unsqueeze(0)
            target = torch.tensor(source[1:min_seq_len + 1]).unsqueeze(0)

            src_batch.append(src)
            target_batch.append(target)

        src_batch = torch.cat(src_batch, dim=0)
        target_batch = torch.cat(target_batch, dim=0)
        return src_batch, target_batch


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


def get_batch(ds, tok, bs, batches_done, L):
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
    idx = starts.unsqueeze(-1) + torch.arange(L)
    combined = torch.gather(sample, 1, idx).to(utils.device)

    return combined
