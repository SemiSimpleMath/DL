import pickle
file_path = "d:/data_sets/probabilistic_transformer/cleaned_simple_wiki_sentences.pkl"
# Load your data
with open(file_path, 'rb') as f:
    data = pickle.load(f)
print(len(data))
# for article in data:
#     for sentence in article:
#         if "||" in sentence:
#             print(article)


# from transformers import GPT2TokenizerFast
# from tokenizers import Tokenizer
#
# tok = Tokenizer.from_file("D:\data_sets\probabilistic_transformer\simple_wikipedia.pkl")
# wrapped_tok = GPT2TokenizerFast(tokenizer_object=tok)
# wrapped_tok.add_tokens(["[UNK]"])
# print("[UNK] token ID:", wrapped_tok.convert_tokens_to_ids("[UNK]"))  # Should not print -100



# from transformers import GPT2Tokenizer
#
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# text = "Here's some example text."
# tokens = tokenizer.tokenize(text)
# print(tokens)
#
# # Get the vocabulary
# vocab = tokenizer.get_vocab()
#
# # Print the vocabulary
# for token, index in vocab.items():
#     print(f"{token}: {index}")

new_data = [' '.join(article) for article in data]

file_path = "d:/data_sets/probabilistic_transformer/cleaned_joined_simple_wiki_sentences.pkl"
# Load your data
with open(file_path, 'wb') as f:
    pickle.dump(new_data, f)