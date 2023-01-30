import config
from datasets import load_dataset

from tokenizers import (
    decoders,
    Tokenizer,
)

from transformers import GPT2TokenizerFast

ds_path = "wikipedia"
ds_file = "20220301.en"
tok_file = "data/tokenizer-wiki-50k.json"

ds = load_dataset(config.ds_path, config.ds_file)
tok = Tokenizer.from_file(tok_file)
tok.decoder = decoders.ByteLevel()

wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tok)

wrapped_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

vocab_size = wrapped_tokenizer.vocab_size + 1
print(vocab_size)
#ds = ds.shuffle(seed=42)

encodes = wrapped_tokenizer.encode("Let's test this tokenizer.")

print(encodes)
print(wrapped_tokenizer.decode(encodes) )

sample = wrapped_tokenizer(
    ds['train'][(8) * 32: (8 + 1) * 32]['text'],
    padding=True,
    truncation=True,
    return_token_type_ids=False,
    return_attention_mask=False,
    return_tensors="pt"
).input_ids



