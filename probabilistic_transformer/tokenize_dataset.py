from tokenizers import Tokenizer

import pickle
file = "d:/data_sets/probabilistic_transformer/cleaned_joined_simple_wiki_sentences.pkl"
with open(file, 'rb') as f:
    data = pickle.load(f)


tok_file = "D:/data_sets/probabilistic_transformer/simple_wikipedia_byte_level.json"
toke = Tokenizer.from_file(tok_file)

tokenized_articles = []
counter = 0
for article in data:
    counter += 1
    tokens = toke.encode(article)
    tokenized_articles.append(tokens.ids)
    if counter % 1000 == 0:
        print(counter)

file = "d:/data_sets/probabilistic_transformer/tokenized_byte_level_simple_wiki_sentences.pkl"
with open(file, "wb") as f:
    pickle.dump(tokenized_articles, f)