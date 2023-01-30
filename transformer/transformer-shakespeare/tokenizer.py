from tokenizers import Tokenizer

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

def tokenizer_train():

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))


    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])


    tokenizer.pre_tokenizer = Whitespace()

    files = [f"./data/wikitext-103-raw-v1/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
    tokenizer.train(files, trainer)

    return tokenizer

def tokenizer_save(tokenizer, path):
    tokenizer.save("data/tokenizer-wiki.json")

def tokenizer_load(path):
    tokenizer = Tokenizer.from_file(path)
    return tokenizer


