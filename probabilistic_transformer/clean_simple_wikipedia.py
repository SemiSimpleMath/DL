import re
from nltk import sent_tokenize
from transformer_libs import tokenizer
import random
random.seed(42)

def load_wikipedia():
    from datasets import load_dataset
    ds = load_dataset("wikipedia", "20220301.en")
    return ds
def load_simple_wikipedia():
    from datasets import load_dataset
    ds = load_dataset("wikipedia", '20220301.simple')
    return ds

def save(sentences):
    with open("./data/simple_wikipedia_sentences.pkl", 'wb') as f:
        pickle.dump(sentences, f)

def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def is_sentence(text: str) -> bool:
    text = text.strip()
    if len(text) < 5:
        return False
    # Check if the text contains at least one letter
    if not re.search(r'[a-zA-Z]', text):
        return False

    # Check if the text starts with a capital letter or a number
    if not re.match(r'^[A-Z0-9]', text):
        return False

    # Check if the text ends with a sentence-ending punctuation mark followed by end of string, or if it is followed by a sentence-ending punctuation followed by space
    if not (re.search(r'[.!?]\s*$', text) or re.search(r'[.!?]\s', text)):
        return False

    # Check if the text contains at least three words
    if len(text.split(" ")) < 3:
        return False

    return True

def create_wiki_sentences(ds):
    article_sentences = []
    total_sentences_start = 0
    total_sentences_end = 0
    article_count = 0
    for article in ds:
        article_count += 1
        if article_count % 100 == 0:
            print(article_count)
        sentences = []  # Initialize a new list of sentences for each article
        lines = article.split("\n")
        for line in lines:
            # Tokenize the line into sentences
            line_sentences = sent_tokenize(line)

            for sentence in line_sentences:
                total_sentences_start += 1
                if is_sentence(sentence):
                    sentences.append(sentence)
                    total_sentences_end += 1

        article_sentences.append(sentences)
    print(f"started with: {total_sentences_start}")
    print(f"ended with: {total_sentences_end}")
    return article_sentences


def create_dataset(ds):
    start = 0
    end = len(ds['train'])
    articles = [ds['train'][i]['text'] for i in range(start, end)]  # Use all the articles
    del ds
    print("Got all articles")
    print("articles before process:" , len(articles))
    articles = create_wiki_sentences(articles)

    # Specify the filename where you want to save the data
    filename = f'd:/data_sets/probabilistic_transformer/articles_simple_wiki_sentences_{start}_{end}.pkl'

    # Use pickle to save the data to the file
    with open(filename, 'wb') as file:
        pickle.dump(articles, file)

    print(f'Saved the data to {filename}')


    return



import pickle
if __name__ == "__main__":

    ds = load_simple_wikipedia()
    print("loaded wikipedia")

    dataset = create_dataset(ds)

