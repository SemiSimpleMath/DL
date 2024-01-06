import collections
import torch
import torch.nn.functional as F
from transformer_libs import utils

# set device to cpu so i can train at same time on the gpu without running
# out of gpu space
device = torch.device("cpu")


def nucleus_sampling(b, p, top_k):
    probs = F.softmax(b, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    sorted_log_probs = torch.log(sorted_probs)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < p
    sorted_log_probs[~nucleus] = float('-inf')
    next_token_indices = torch.multinomial(probs, top_k, replacement=False)
    return next_token_indices


def beam_search_2(model, params, width, max_depth, p_nuc, prompt, tok=None):
    d_model = params['d_model']
    node = {'prompt': prompt, 'score': 0, 'tokens': [], 'depth': 0}
    candidates = []
    final_candidates = []
    q = collections.deque()
    q.append(node)

    while len(q) > 0:
        node = q.pop()

        if node['depth'] == max_depth:
            tokens = node['tokens']
            tokens = [str(t) for t in tokens]
            tok_str = " ".join(tokens)
            list_prompt = prompt[0].tolist()
            list_prompt = [str(t) for t in list_prompt]
            str_prompt = " ".join(list_prompt)
            count = str_prompt.count(tok_str)
            node['score'] += node['score'] * count
            final_candidates.append(node)
            continue
        prompt = node['prompt']
        seq_len = prompt.shape[-1]

        out, _ = model(prompt)
        last_row = out[:, -1, :]
        last_row = last_row.squeeze()
        children = nucleus_sampling(last_row, p_nuc, width)  # this is token indices and their corresponding probs
        probs = F.softmax(last_row, dim=-1)
        for c in children:
            index = c.item()
            #dip = probs[9440]
            token = index
            score = -torch.log(probs[index])
            #debug = tok.decode(token)
            #val = probs[index]
            # TODO implement a penalty for repeating tokens in a row. For now model seems pretty good
            # about not repeating.

            # pl = prompt.tolist()
            # pl = pl[0]
            #ct = node['tokens'].count(token)
            #score *= ct
            child_data = {'score': node['score'] + score, 'token': token, 'tokens': node['tokens'] + [token]}
            token = torch.ones(1) * token
            child_data['prompt'] = torch.cat([node['prompt'].squeeze(), token], -1).to(torch.int64).unsqueeze(0)
            child_data['depth'] = node['depth'] + 1
            candidates.append(child_data)

        if len(q) == 0 and len(candidates) > 0:
            top_width_candidates = sorted(candidates, key=lambda d: d['score'])[:width]

            for c in top_width_candidates:
                q.append(c)
            candidates = []
            continue

    winner = sorted(final_candidates, key=lambda d: d['score'])[0]

    # get the top candidate and its first word.  This is what we return!

    next_token, top_tokens = winner['tokens'][0], sorted(final_candidates, key=lambda d: d['score'])

    return next_token, [t['tokens'][0] for t in top_tokens]


def beam_search(model, params, width, p, prompt, n):
    d_model = params['d_model']

    lookup = {}

    # Let's assume we start with single prompt sequence, and we are trying to predict next word

    # for however many tokens we want to generate
    for i in range(n):
        # generate the batches
        seq_len = prompt.shape[-1]

        b = model(prompt)

        b = b[:, -1, :]
        candidates = []
        for seq_num, seq in enumerate(b):  # Here b must be either 1xd_vocab or its beam_width x d_vocab
            top_k = nucleus_sampling(seq, p, width)  # top_k is 1 x beam_width has indices
            top_k = top_k.squeeze()
            for index in top_k:
                index = index.item()
                token = index
                seq = seq.squeeze()
                seq = F.softmax(seq, dim=-1)
                score = seq[index]

                # if this is the first iteration write into the table row # and score
                # add the word to a list if we want.

                candidate = {'row': seq_num, 'token': token}

                if i == 0:
                    lookup[seq_num] = {}
                    lookup[seq_num]['score'] = -torch.log(score)
                    lookup[seq_num]['token'] = token
                    candidate['score'] = torch.log(score)
                else:

                    candidate['score'] = lookup[seq_num]['score'] - torch.log(score)

                candidates.append(candidate)

        # once we have processed all the rows we take top width many choices from the list
        top_width_candidates = sorted(candidates, key=lambda d: d['score'])[:width]

        lookup = {}  # dump contents of look up

        for j, c in enumerate(top_width_candidates):
            lookup[j] = {}
            lookup[j]['score'] = c['score']
            lookup[j]['token'] = c['token']

        # these words are then used to create the next batch

        prompt = generate_new_beam_batch(prompt, top_width_candidates)

    return prompt


def generate_new_beam_batch(prev_batch, new_beam):
    new_rows = []
    for nb in new_beam:
        row = nb['row']
        token = nb['token']

        old_row = prev_batch[row]

        token = torch.ones(1) * token
        token = token.to(device)
        new_row = torch.cat([old_row.clone(), token], -1)

        new_rows.append(new_row.unsqueeze(0))

    new_batch = torch.cat(new_rows, dim=0)
    new_batch = new_batch.long()

    return new_batch


def generate_next_n_tokens(prompt, n, search_function, model, model_params, width, max_depth, p_nuc, tok):
    top_toks = []
    for _ in range(n):
        result, top_tokens = search_function(model, model_params, width, max_depth, p_nuc, prompt, tok)
        top_toks.append((tok.decode(result), [tok.decode(t) for t in top_tokens]))
        #print("------------------------------------------------------")
        #print(tok.decode(result))
        print(tok.decode(result), end="")
        result = torch.ones(1) * result
        prompt = torch.cat([prompt.squeeze(), result], -1).to(torch.int64).unsqueeze(0)
        #print(result)
        #print(prompt)
        #debug_check = prompt.tolist()
        #debug_p = tok.decode(debug_check[0])

        #print(debug_p)

    return prompt, top_toks
