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
    width = 10
    p_nuc = .95
    max_depth = 2
    n = 100

    # Load the tokenizer
    tok_file = config.tok_file
    tok = tokenizer.WikiTok()
    tok.load(tok_file)

    prompt = "When I was older we moved to Paris"
    print(f'Prompt: {prompt}')

    text_prompt = prompt
    prompt = tok.text_to_tensor(prompt)
    prompt = prompt.unsqueeze(0)
    prompt = prompt.to(device)

    file = utils.most_recent_file(config.model_directory)
    model, model_params = utils.load_model_inference(file, False)
    model.eval()

    # utils.to_cuda(model) # It is often preferable not to have the model run on cuda for inference.

    print(text_prompt, end=" ")
    prompt, result_tokens = beam_search_lib.generate_next_n_tokens(prompt, n, beam_search_lib.beam_search_2, model,
                                                                   model_params, width, max_depth, p_nuc, tok)

main()
