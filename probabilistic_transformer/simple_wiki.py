import string
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize (split) the text into words
    words = text.split()
    return words

import pickle
file = "d:/data_sets/probabilistic_transformer/articles_simple_wiki_sentences_0_205328.pkl"
with open(file, "rb") as f:
    data = pickle.load(f)

# unique_words = set()
# for article in data:
#     for sentence in article:
#         words = preprocess_text(sentence)
#         unique_words.update(words)
#
# print(len(unique_words)) # 520256


test_data = data
total_sentences = 0
total_cleaned_sentences = 0
cleaned_articles = []
for article in test_data:
    cleaned_article = []
    for sentence in article:
        total_sentences += 1
        non_alpha = sum((not char.isalpha() and char !=' ') for char in sentence)
        total_chars = len(sentence)

        if total_chars == 0:
            continue

        non_alpha_percentage = (non_alpha / total_chars) * 100

        if non_alpha_percentage < 15:
            total_cleaned_sentences += 1
            cleaned_article.append(sentence)
    cleaned_articles.append(cleaned_article)

print(cleaned_articles)
print(total_sentences)
print(total_cleaned_sentences)
file = "d:/data_sets/probabilistic_transformer/cleaned_simple_wiki_sentences.pkl"
with open(file, "wb") as f:
    pickle.dump(data, f)

