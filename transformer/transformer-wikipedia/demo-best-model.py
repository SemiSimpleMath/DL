import torch
import config_best_model
from transformer_libs import data_utils
from transformer_libs import utils
from transformer_libs import tokenizer
from transformer_libs import beam_search_lib
from transformer_libs import decoder
# set device to cpu so i can train at same time on the gpu without running
# out of gpu space
device = torch.device("cpu")


def main():
    width = 3
    p_nuc = .1
    max_depth = 2
    n = 200

    # Load the tokenizer
    tok = tokenizer.load_tokenizer(config_best_model.tokenizer_class, config_best_model.tok_file, config_best_model.tok_special_tokens)
    vocab_size = tok.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    encoded, decoded = tok.test("Testing the tokenizer!")
    print(encoded, decoded)


    prompt = "One of the best modern painters is "
    print(f'Prompt: {prompt}')

    text_prompt = prompt
    prompt = tok.text_to_tensor(prompt)
    prompt = prompt.unsqueeze(0)
    prompt = prompt.to(device)

    file = utils.most_recent_file(config_best_model.model_directory)
    model_class = decoder.Decoder
    model, optimizer, model_params, train_params, lr_params = utils.load_old_model(model_class, file, device=device)
    model.eval()

    # utils.to_cuda(model) # It is often preferable not to have the model run on cuda for inference.

    print(text_prompt, end=" ")
    prompt, result_tokens = beam_search_lib.generate_next_n_tokens(prompt, n, beam_search_lib.beam_search_2, model,
                                                                   model_params, width, max_depth, p_nuc, tok)



main()
