import torch
import config
from transformer_libs import data_utils
from transformer_libs import utils
from transformer_libs import tokenizer
from transformer_libs import beam_search_lib


# set device to cpu so i can train at same time on the gpu without running
# out of gpu space
device = torch.device("cpu")


def main():
    width = 3
    p_nuc = .90
    max_depth = 2
    n = 80
    # Load the tokenizer
    tok_file = config.tok_file
    tok = tokenizer.load_tokenizer(tok_file)

    prompt = "Jukka Virtanen is a Finnish"
    print(f'Prompt: {prompt}')
    text_prompt = prompt
    prompt = data_utils.text_to_model_input(prompt, tok)
    prompt = prompt.unsqueeze(0)
    prompt = prompt.to(device)

    # Get the vocab size. +1 is due to [PAD] token which we force to be the last token
    vocab_size = tok.vocab_size + 1

    file = utils.most_recent_file(config.model_directory)
    model, model_params = utils.load_model_inference(file, False)
    model.eval()
    # utils.to_cuda(model)
    print(text_prompt, end=" ")
    result_tokens = beam_search_lib.generate_next_n_tokens(prompt, n,beam_search_lib.beam_search_2, model, model_params, width, max_depth, p_nuc, tok)


main()
