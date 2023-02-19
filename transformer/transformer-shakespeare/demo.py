import torch
import config
from transformer_libs import data_utils
from transformer_libs import utils
from transformer_libs import beam_search_lib
from transformer_libs import tokenizer

# set device to cpu so i can train at same time on the gpu without running
# out of gpu space
device = torch.device("cpu")

def main():
    width = 3
    p_nuc = .90
    max_depth = 3
    n = 100

    torch.manual_seed(0)

    # Load the tokenizer
    tok_file = config.tok_file
    tok = tokenizer.ShakespeareTok()
    tok.load(tok_file)


    prompt = "To be or not to be"
    print(f'Prompt: {prompt}')


    text_prompt = prompt
    prompt = tok.text_to_tensor(prompt)
    prompt = prompt.unsqueeze(0)
    prompt = prompt.to(device)

    file = utils.most_recent_file(config.model_directory)
    model, model_params = utils.load_model_inference(file, False)
    model.eval()

    print(text_prompt, end=" ")
    prompt, result_tokens = beam_search_lib.generate_next_n_tokens(prompt, n, beam_search_lib.beam_search_2, model, model_params, width, max_depth, p_nuc, tok)


main()


