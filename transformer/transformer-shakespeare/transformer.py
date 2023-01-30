
import torch
import torch.nn as nn
import decoder
import utils
import time
import tokenizer
import data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model):
    PATH = '.\models\model' + time.strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), PATH)

def train(module, opt, loss, bs, num_batches, d_model, DL, seq_len, pos_enc, mask):

    total_loss = 0
    output_every = 500
    total_sequences = 0

    for batch_num in range(num_batches):
        src, target = DL.get_batch(bs)

        src = src.to(device)
        target = target.to(device)

        pe = pos_enc[0,:seq_len,:d_model]
        msk = mask[0, :seq_len,: seq_len]

        pred = module(src, pe, msk)
        pred = pred.permute(0,2,1)

        total_sequences += bs * seq_len

        # Compute the loss
        l = loss(pred, target)
        total_loss += l.item()

        # Backward pass
        l.backward()

        # Update the parameters
        opt.step()
        opt.zero_grad()

        if batch_num % output_every == 0 and batch_num != 0:
            print(total_loss/output_every)
            print(l)
            total_loss = 0
            print("total sequences: ", total_sequences)


def main():
    tokenizer_path = "data/tokenizer-wiki.json"
    tok = tokenizer.tokenizer_load(tokenizer_path)

    print("vocab before shakespeare", tok.get_vocab_size())

    shakespeare_path = 'data/shakespeare/shakespeare.csv'
    s_data = data_utils.read_shakespeare_data(shakespeare_path)

    tok.add_tokens(list(s_data))
    shakespeare_str = ' '.join(s_data)
    tokenized_shakespeare = tok.encode(shakespeare_str)

    print("vocab after shakespeare: ", tok.get_vocab_size())

    # Model paramaters
    num_blocks = 6
    d_model = 256
    d_middle = 4 * d_model
    vocab_size = tok.get_vocab_size() + 1
    dropout = 0.1
    h = 6
    d_Q = d_model
    d_K = d_model
    d_V = d_model
    MAX_SEQ_LEN = 100
    pos_enc = decoder.positional_encoding(MAX_SEQ_LEN, d_model)
    print(pos_enc.shape)
    pos_enc = pos_enc.to(device)
    mask = decoder.create_upper_mask(MAX_SEQ_LEN + 2)
    mask = mask.to(device)

    print(mask)

    seq_len = 100

    pe = pos_enc[0,:seq_len+1,:d_model]
    msk = mask[0, :seq_len+1,: seq_len+1]
    print(msk)

    print(pe.shape)


    pe = pe.to(device)
    msk = msk.to(device)


    shakespeare_ids = torch.LongTensor(tokenized_shakespeare.ids)

    DL = data_utils.DataLoader(shakespeare_ids, seq_len)

    module = decoder.Decoder(num_blocks, d_model, d_middle, vocab_size, dropout, h, d_Q, d_K, d_V)

    LOAD = True
    if LOAD:
        PATH = 'models/model20230115-190243'
        module.load_state_dict(torch.load(PATH))


    utils.to_cuda(module)

    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(module.parameters(), lr=2.5e-4)
    bs = 32
    num_batches = 10000
    #train(module, opt, loss, bs, num_batches, d_model, DL, seq_len, pos_enc, mask)
    #save_model(module)

main()