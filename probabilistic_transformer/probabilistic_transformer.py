import torch
import os
import config
import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch.nn as nn
import random
import pickle
from transformer_libs import tokenizer
from transformer_libs import data_utils
from transformer_libs import utils
from transformer_libs import train


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)

import torch



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        pos = self.encoding[:, :x.size(1)].detach().to(device)
        return x + pos


def probabilistic_transformer_loss_1(preds, target, beta=1.0, sigma_eps=1e-5, sigma_max_penalty=0.1,
                                     semantic_loss_weight=0.5):
    output_token, mu_out, sigma_out = preds
    target_token, target_mu = target

    # target_sigma needs to be computed.
    # Compute the token classification loss
    classification_loss = F.cross_entropy(output_token.view(-1, output_token.size(-1)), target_token.view(-1))

    # Compute the semantic similarity between predicted mu and target mu
    semantic_similarity = F.cosine_similarity(mu_out, target_mu, dim=-1).mean()

    # Reduce the penalty for classification loss based on semantic similarity
    adjusted_classification_loss = (1 - semantic_loss_weight * semantic_similarity) * classification_loss

    # Compute the cosine similarity loss for mu
    mu_loss = 1 - semantic_similarity

    abs_diff = torch.abs(mu_out - target_mu)

    # Compute the mean absolute deviation across the batch dimension
    approx_sigma = torch.mean(abs_diff, dim=0) + sigma_eps

    # Ensure approx_sigma has the same shape as sigma_out for loss calculation
    approx_sigma = approx_sigma.expand_as(sigma_out)

    # Compute the loss for sigma, encouraging it to be close to the approximated sigma
    sigma_loss = F.mse_loss(sigma_out, approx_sigma)

    # Add a penalty for very small values of sigma
    sigma_penalty = torch.where(sigma_out < sigma_eps, sigma_max_penalty * torch.ones_like(sigma_out),
                                torch.zeros_like(sigma_out))
    sigma_loss += sigma_penalty.mean()

    # Total loss is a weighted sum of the classification loss and the probabilistic losses
    total_loss = adjusted_classification_loss + beta * (mu_loss + sigma_loss)
    return total_loss , adjusted_classification_loss, mu_loss, sigma_loss, semantic_similarity


def probabilistic_transformer_loss_2(preds, target, alpha=1.0, beta=1.0, semantic_loss_weight=0.5):
    output_token, mu_out, sigma_out = preds
    target_token, target_mu = target

    # Compute the token classification loss
    classification_loss = F.cross_entropy(output_token.view(-1, output_token.size(-1)), target_token.view(-1))

    # Compute the semantic similarity between predicted mu and target mu
    semantic_similarity = F.cosine_similarity(mu_out, target_mu, dim=-1).mean()

    # Reduce the penalty for classification loss based on semantic similarity
    #adjusted_classification_loss = (1 - semantic_loss_weight * semantic_similarity) * classification_loss
    k = 10  # Steepness of the transition
    b = 0.6  # Point of transition
    semantic_loss_weight = 1 / (1 + torch.exp(-k * (semantic_similarity - b)))

    adjusted_classification_loss = (1 - semantic_loss_weight) * classification_loss
    # Compute the cosine similarity loss for mu
    mu_loss = 1 - semantic_similarity

    # Penalty for being confident but wrong
    penalty = (mu_loss ** 2) / (sigma_out.mean() ** 2 + 1e-5)  # adding a small epsilon for numerical stability

    # Penalize large values of sigma
    sigma_penalty = torch.exp(alpha * sigma_out.mean())

    # Total loss is a weighted sum of the classification loss, mu loss, sigma penalty, and penalty
    total_loss = adjusted_classification_loss + mu_loss + beta * (sigma_penalty + penalty)

    return total_loss, adjusted_classification_loss, mu_loss, semantic_similarity


class ProbabilisticEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_mu = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_sigma = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids):
        mu = self.embedding_mu(input_ids)
        sigma = F.softplus(self.embedding_sigma(input_ids))  # Ensure sigma is positive
        return mu, sigma


import torch.nn as nn
import torch.nn.functional as F


class ProbabilisticTokenTransformer(nn.Module):
    def __init__(self, embedding_dim, d_model):
        super().__init__()
        self.query_mu = nn.Linear(embedding_dim, d_model)
        self.query_sigma = nn.Linear(embedding_dim, d_model)
        self.key_mu = nn.Linear(embedding_dim, d_model)
        self.key_sigma = nn.Linear(embedding_dim, d_model)
        self.value_mu = nn.Linear(embedding_dim, d_model)
        self.value_sigma = nn.Linear(embedding_dim, d_model)

    def forward(self, probabilistic_token):
        mu, sigma = probabilistic_token

        query_mu = self.query_mu(mu)
        query_sigma = F.softplus(self.query_sigma(sigma))

        key_mu = self.key_mu(mu)
        key_sigma = F.softplus(self.key_sigma(sigma))

        value_mu = self.value_mu(mu)
        value_sigma = F.softplus(self.value_sigma(sigma))

        return (query_mu, query_sigma), (key_mu, key_sigma), (value_mu, value_sigma)


class ProbabilisticAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scaling_factor = (self.d_k) ** 0.5
        self.beta = 0.5
        # Define separate linear layers for sigma and mu of Q, K, and V
        self.q_linear_mu = nn.Linear(d_model, d_model)
        self.k_linear_mu = nn.Linear(d_model, d_model)
        self.v_linear_mu = nn.Linear(d_model, d_model)
        self.q_linear_sigma = nn.Linear(d_model, d_model)
        self.k_linear_sigma = nn.Linear(d_model, d_model)
        self.v_linear_sigma = nn.Linear(d_model, d_model)

    def bhattacharyya_distance(self, mu_q, sigma_q, mu_k, sigma_k, epsilon=1e-6):
        term_1 = 0.25 * torch.log(0.25 * ((sigma_q ** 2 + epsilon) / (sigma_k ** 2 + epsilon) +
                                          (sigma_k ** 2 + epsilon) / (sigma_q ** 2 + epsilon) + 2))
        term_2 = 0.25 * ((mu_q - mu_k) ** 2) / (sigma_q ** 2 + sigma_k ** 2 + epsilon)
        return term_1 + term_2


    def forward(self, query, key, value, query_sigma, key_sigma, value_sigma, mask=None, epsilon=1e-6):
        batch_size, seq_len, _ = query.size()

        # Use the correct linear layer for sigma and mu
        mu_q = self.q_linear_mu(query).view(batch_size, seq_len, self.nhead, self.d_k)
        sigma_q = self.q_linear_sigma(query_sigma).view(batch_size, seq_len, self.nhead, self.d_k)
        mu_k = self.k_linear_mu(key).view(batch_size, seq_len, self.nhead, self.d_k)
        sigma_k = self.k_linear_sigma(key_sigma).view(batch_size, seq_len, self.nhead, self.d_k)
        mu_v = self.v_linear_mu(value).view(batch_size, seq_len, self.nhead, self.d_k)
        sigma_v = self.v_linear_sigma(value_sigma).view(batch_size, seq_len, self.nhead, self.d_k)
        # Compute Bhattacharyya distances
        mu_q, sigma_q = mu_q.unsqueeze(3), sigma_q.unsqueeze(3)
        mu_k, sigma_k = mu_k.unsqueeze(2), sigma_k.unsqueeze(2)
        distances = self.bhattacharyya_distance(mu_q, sigma_q, mu_k, sigma_k).sum(dim=-1) / self.scaling_factor
        assert not torch.isnan(distances).any()
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add head dimension
            distances = distances.masked_fill(mask == 0, float('-inf'))

        # Compute attention scores
        attn = F.softmax(-self.beta * distances, dim=-1)


        # Apply attention to means
        output_mu = torch.matmul(attn, mu_v)
        assert not torch.isnan(output_mu).any()

        # Propagate uncertainties: Compute weighted sum of variances, then take square root
        var_v = sigma_v ** 2
        output_var = torch.clamp(torch.matmul(attn, var_v), min=0)
        output_sigma = torch.sqrt(output_var + epsilon)
        # Concatenate heads
        output_mu = output_mu.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output_sigma = output_sigma.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        assert not torch.isnan(output_mu).any()
        assert not torch.isnan(output_sigma).any()
        return output_mu, output_sigma


class ProbabilisticTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dim_out):
        super().__init__()
        self.attention = ProbabilisticAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_out)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(dim_out)
        self.dropout = nn.Dropout(0.1)
        self.probabilistic_token_transformer = ProbabilisticTokenTransformer(d_model, d_model)

    def forward(self, mu, sigma):
        # Transform tokens into means and standard deviations for Q, K, V
        (query, query_sigma), (key, key_sigma), (value, value_sigma) = self.probabilistic_token_transformer((mu, sigma))

        # Ensure query_sigma, key_sigma, and value_sigma are positive
        query_sigma = F.softplus(query_sigma)
        key_sigma = F.softplus(key_sigma)
        value_sigma = F.softplus(value_sigma)

        # Compute attention, separately handling means and standard deviations
        attn_output_mu, attn_output_sigma = self.attention(query, key, value, query_sigma, key_sigma, value_sigma)

        # Skip connection for means
        mu = self.norm1(mu + attn_output_mu)

        # Skip connection for standard deviations
        sigma = self.norm1(torch.sqrt(sigma ** 2 + attn_output_sigma ** 2))

        # Feed-forward for means (with ReLU and dropout)
        ff_output_mu = F.relu(self.linear1(mu))
        ff_output_mu = self.dropout(self.linear2(ff_output_mu))

        # Propagate standard deviations through feedforward layers
        ff_output_sigma = F.softplus(self.linear2(self.linear1(sigma)))

        # Skip connection for means and standard deviations
        mu = self.norm2(mu + ff_output_mu)
        sigma = self.norm2(torch.sqrt(sigma ** 2 + ff_output_sigma ** 2))

        return mu, sigma


class ProbabilisticTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embedding = ProbabilisticEmbedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            ProbabilisticTransformerBlock(d_model, num_heads, 2 * d_model, d_model)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.out_mu = nn.Linear(d_model, d_model)
        self.out_sigma = nn.Linear(d_model, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, input_ids):
        input_ids, _ = input_ids
        mu, sigma = self.embedding(input_ids)
        mu = self.positional_encoding(mu)
        for block in self.transformer_blocks:
            mu, sigma = block(mu, sigma)
        logits = self.output_projection(mu) # should this not depend on both mu and sigma?
        mu_out = self.out_mu(mu)
        sigma_out = F.softplus(self.out_sigma(sigma))
        return logits, mu_out, sigma_out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(model_params, train_params, lr_params):
    print('Generating a new model to train.')
    vocab_size = model_params['vocab_size']
    embedding_dim = model_params['d_model']
    num_heads = model_params['h']
    num_layers = model_params['num_blocks']
    model = ProbabilisticTransformer(vocab_size, embedding_dim, num_heads, num_layers)
    train_params['batch_num'] = 0
    train_params['samples_done'] = 0
    opt = torch.optim.Adam(model.parameters(), lr=lr_params['lr'])
    return model, opt, model_params, train_params, lr_params


def custom_loss():
    return None


def prepare_train_config():
    model_params = config.model_params
    lr_params = config.lr_params
    train_params = config.train_params
    config_params = config.config_params
    return model_params, train_params, lr_params, config_params


def main():
    torch.manual_seed(0)

    # Load the tokenizer
    tok_file = config.tok_file
    tok = tokenizer.WikiTok()
    tok.load(tok_file)

    # Demo the tokenizer
    tok.tokenizer_test()

    vocab_size = tok.get_vocab_size()

    # Load the dataset
    ds = data_utils.load_ds(config.wiki_ds_file)
    random.shuffle(ds)

    print("Total GPU Memory: ", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
    print("Reserved Memory: ", torch.cuda.memory_reserved(0) / 1e9, "GB")
    print("Allocated Memory: ", torch.cuda.memory_allocated(0) / 1e9, "GB")

    vocab_size = tok.get_vocab_size()

    model_params, train_params, lr_params, config_params = prepare_train_config()
    model_params['vocab_size'] = vocab_size

    model, opt, model_params, train_params, lr_params = create_model(model_params, train_params, lr_params)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    utils.print_params(model_params)
    utils.print_params(train_params)
    utils.print_params(lr_params)
    utils.print_params(config_params)

    # Get the total number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {pytorch_total_params}')

    # Set the loss function as cross-entropy loss
    loss_func = probabilistic_transformer_loss_2
    # lr_func = utils.constant_lr
    # lr_scale_factor = .1
    lr_func = utils.get_lr
    model_params['samples_done'] = 0
    model_params['batch_num'] = 0
    model_params['id'] = 'prob2'

    start = 0
    end = 50000
    with open(f'd:/data_sets/probabilistic_transformer/wikipedia/tokenized_articles_{start}_{end}.pkl', 'rb') as f:
        ds = pickle.load(f)
    print("loaded train data")
    file = "D:/data_sets/probabilistic_transformer/token_to_embedding_dict.pkl"
    with open(file, 'rb') as f:
        embedding_dict = pickle.load(f)
    print("loaded embedding dict")

    dl_params = {'L': model_params['seq_len'], 'tok': tok, 'ds': ds, 'article_idx': 0, 'start': 0,
                 'embedding_dict': embedding_dict, }
    dl = data_utils.ProbabilisticDL(model_params['bs'], model_params['samples_done'], dl_params)

    trainer = train.Train(model, opt, model_params, train_params, lr_params, config_params, loss_func, dl,
                          lr_func)

    trainer.train(50_000)


import fasttext.util
import os

# Load the pre-trained FastText model (English)
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

import scipy.stats


# Function to convert an embedding to a distribution
def embedding_to_distribution(embedding, variance=0.01):
    covariance_matrix = np.eye(len(embedding)) * variance
    distribution = scipy.stats.multivariate_normal(mean=embedding, cov=covariance_matrix)
    return distribution


# Function to compute KL divergence between two multivariate Gaussian distributions

import numpy as np

if __name__ == "__main__":
    import os

    print("Current Working Directory:", os.getcwd())

    # Get embeddings for "cat" and "dog"
    embedding_happy = ft.get_word_vector('happiness')
    embedding_joyful = ft.get_word_vector('sadness')

    from sklearn.metrics.pairwise import cosine_similarity

    # Assume embedding_happy and embedding_joyful are the embeddings you obtained from FastText

    # Normalize the vectors to unit length
    embedding_happy_normalized = embedding_happy / np.linalg.norm(embedding_happy)
    embedding_joyful_normalized = embedding_joyful / np.linalg.norm(embedding_joyful)

    # Calculate cosine similarity
    cos_sim = np.dot(embedding_happy_normalized, embedding_joyful_normalized)

    print('Cosine Similarity:', cos_sim)

    main()



    # def bhattacharyya_distance(self, mu1, sigma1, mu2, sigma2, epsilon=1e-6):
    #     # Ensure covariance matrices are positive definite
    #     sigma1 = sigma1 + torch.eye(sigma1.size(-1)) * epsilon
    #     sigma2 = sigma2 + torch.eye(sigma2.size(-1)) * epsilon
    #
    #     # Compute the mean of the covariance matrices
    #     sigma = 0.5 * (sigma1 + sigma2)
    #
    #     # Compute the Bhattacharyya distance
    #     term_1 = 0.125 * (mu2 - mu1).matmul(torch.inverse(sigma)).matmul(mu2 - mu1)
    #     term_2 = 0.5 * torch.log(torch.det(sigma) / (torch.sqrt(torch.det(sigma1) * torch.det(sigma2))))
    #
    #     distance = term_1 + term_2
    #     return distance
