
def demo_model(model):
    model.eval()
    MAX_SEQ_LEN = 50
    d_model = 12
    bs = 1
    seq_len = random.randint(1, MAX_SEQ_LEN)
    enc_src, dec_src, target = generate_batch(bs, seq_len, d_model)
    original_seq = enc_src
    enc_src = F.one_hot(enc_src.to(torch.int64), num_classes=d_model).float()
    dec_src = F.one_hot(dec_src.to(torch.int64), num_classes=d_model).float()

    enc_src = enc_src.to(device)
    dec_src = dec_src.to(device)
    target = target.to(device)

    pos_enc = positional_encoding(MAX_SEQ_LEN, d_model)
    pos_enc = pos_enc.to(device)
    mask = create_upper_mask(MAX_SEQ_LEN + 2)
    mask = mask.to(device)

    pe = pos_enc[0,:seq_len+1,:d_model]
    msk = mask[0, :seq_len+1,: seq_len+1]
    pred = model(enc_src, dec_src, pe, None, msk)

    pred = pred.permute(0,2,1)
    pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
    print(pred.shape)
    print(target.shape)
    print(pred)
    print(target)
    print(original_seq)