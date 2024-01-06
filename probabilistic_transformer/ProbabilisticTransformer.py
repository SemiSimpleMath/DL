import torch
import os
import config
import numpy as np

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = nn.Parameter(self.encoding.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        pos = self.encoding[:, :x.size(1)].detach().to(x.device)
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
    return total_loss, adjusted_classification_loss, sigma_loss, semantic_similarity


def probabilistic_transformer_loss_2(preds, target, alpha=1.0, beta=1.0, semantic_loss_weight=0.5):
    output_token, mu_out, sigma_out = preds
    target_token, target_mu = target

    # Compute the token classification loss
    classification_loss = F.cross_entropy(output_token.view(-1, output_token.size(-1)), target_token.view(-1))

    # Compute the semantic similarity between predicted mu and target mu
    semantic_similarity = F.cosine_similarity(mu_out, target_mu, dim=-1).mean()

    # Reduce the penalty for classification loss based on semantic similarity
    # adjusted_classification_loss = (1 - semantic_loss_weight * semantic_similarity) * classification_loss
    k = 10  # Steepness of the transition
    b = 0.7  # Point of transition
    semantic_loss_weight = 1 / (1 + torch.exp(-k * (semantic_similarity - b)))

    adjusted_classification_loss = (1 - semantic_loss_weight) * classification_loss/ 10
    # Compute the cosine similarity loss for mu
    mu_loss = 1 - semantic_similarity

    # Penalty for being confident but wrong
    penalty = (mu_loss ** 2) / (sigma_out.mean() ** 2 + 1e-5)  # adding a small epsilon for numerical stability

    # Penalize large values of sigma
    sigma_penalty = torch.exp(alpha * sigma_out.mean())

    # Total loss is a weighted sum of the classification loss, mu loss, sigma penalty, and penalty
    total_loss = adjusted_classification_loss + mu_loss + beta * (sigma_penalty + penalty)

    return total_loss, adjusted_classification_loss, mu_loss, semantic_similarity


def probabilistic_transformer_loss_3(preds, target):
    output_token, mu_out, sigma_out = preds
    target_token, target_mu = target
    loss = F.cross_entropy(output_token.view(-1, output_token.size(-1)), target_token.view(-1))
    return loss, None, None, None


class ProbabilisticEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_mu = nn.Linear(vocab_size, embedding_dim)
        self.embedding_sigma = nn.Linear(vocab_size, embedding_dim)

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
        self.beta = 0.5  # self.beta = nn.Parameter(torch.tensor(0.5))
        # Define separate linear layers for sigma and mu of Q, K, and V
        self.q_linear_mu = nn.Linear(d_model, d_model)
        self.k_linear_mu = nn.Linear(d_model, d_model)
        self.v_linear_mu = nn.Linear(d_model, d_model)
        self.q_linear_sigma = nn.Linear(d_model, d_model)
        self.k_linear_sigma = nn.Linear(d_model, d_model)
        self.v_linear_sigma = nn.Linear(d_model, d_model)

    def bhattacharyya_distance(self, mu_q, sigma_q, mu_k, sigma_k, epsilon=1e-6):
        bs, seq_len_q, num_heads, feature_dim_per_head = mu_q.shape
        _, seq_len_k, _, _ = mu_k.shape

        # Expand dimensions to calculate pair-wise distances between all elements in queries and keys
        mu_q = mu_q.unsqueeze(2)  # [bs, seq_len_q, 1, num_heads, feature_dim_per_head]
        sigma_q = sigma_q.unsqueeze(2)  # [bs, seq_len_q, 1, num_heads, feature_dim_per_head]
        mu_k = mu_k.unsqueeze(1)  # [bs, 1, seq_len_k, num_heads, feature_dim_per_head]
        sigma_k = sigma_k.unsqueeze(1)  # [bs, 1, seq_len_k, num_heads, feature_dim_per_head]

        # Compute Bhattacharyya distance
        term_1 = 0.25 * torch.log(0.25 * ((sigma_q ** 2 + epsilon) / (sigma_k ** 2 + epsilon) +
                                          (sigma_k ** 2 + epsilon) / (sigma_q ** 2 + epsilon) + 2))
        term_2 = 0.25 * ((mu_q - mu_k) ** 2) / (sigma_q ** 2 + sigma_k ** 2 + epsilon)

        return (term_1 + term_2).sum(-1)  # Sum over the last dimension and return [bs, seq_len_q, seq_len_k, num_heads]

    # def forward(self, query, key, value, query_sigma, key_sigma, value_sigma, mask=None, epsilon=1e-6):
    #     batch_size, seq_len, _ = query.size()
    #
    #     # Use the correct linear layer for sigma and mu
    #     mu_q = self.q_linear_mu(query).view(batch_size, seq_len, self.nhead, self.d_k)
    #     sigma_q = self.q_linear_sigma(query_sigma).view(batch_size, seq_len, self.nhead, self.d_k)
    #     mu_k = self.k_linear_mu(key).view(batch_size, seq_len, self.nhead, self.d_k)
    #     sigma_k = self.k_linear_sigma(key_sigma).view(batch_size, seq_len, self.nhead, self.d_k)
    #     mu_v = self.v_linear_mu(value).view(batch_size, seq_len, self.nhead, self.d_k)
    #     sigma_v = self.v_linear_sigma(value_sigma).view(batch_size, seq_len, self.nhead, self.d_k)
    #     # Compute Bhattacharyya distances
    #     mu_q, sigma_q = mu_q.unsqueeze(3), sigma_q.unsqueeze(3)
    #     mu_k, sigma_k = mu_k.unsqueeze(2), sigma_k.unsqueeze(2)
    #     distances = self.bhattacharyya_distance(mu_q, sigma_q, mu_k, sigma_k).sum(dim=-1) / self.scaling_factor
    #     assert not torch.isnan(distances).any()
    #     if mask is not None:
    #         mask = mask.unsqueeze(1)  # Add head dimension
    #         distances = distances.masked_fill(mask == 0, float('-inf'))
    #
    #     # Compute attention scores
    #     attn = F.softmax(-self.beta * distances, dim=-1)
    #
    #     # Apply attention to means
    #     output_mu = torch.matmul(attn, mu_v)
    #
    #     # Propagate uncertainties: Compute weighted sum of variances, then take square root
    #     var_v = sigma_v ** 2
    #     output_var = torch.clamp(torch.matmul(attn, var_v), min=0)
    #     output_sigma = torch.sqrt(output_var + epsilon)
    #     # Concatenate heads
    #     output_mu = output_mu.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    #     output_sigma = output_sigma.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    #    return output_mu, output_sigma

    def forward(self, query, key, value, query_sigma, key_sigma, value_sigma, mask=None, epsilon=1e-6):
        batch_size, seq_len, _ = query.size()

        # Split into multiple heads
        mu_q = self.q_linear_mu(query).view(batch_size, seq_len, self.nhead, self.d_k)
        mu_k = self.k_linear_mu(key).view(batch_size, seq_len, self.nhead, self.d_k)
        mu_v = self.v_linear_mu(value).view(batch_size, seq_len, self.nhead, self.d_k)

        sigma_q = self.q_linear_sigma(query_sigma).view(batch_size, seq_len, self.nhead, self.d_k)
        sigma_k = self.k_linear_sigma(key_sigma).view(batch_size, seq_len, self.nhead, self.d_k)
        sigma_v = self.v_linear_sigma(value_sigma).view(batch_size, seq_len, self.nhead, self.d_k)

        # Compute Bhattacharyya distances
        distances = self.bhattacharyya_distance(mu_q, sigma_q, mu_k, sigma_k) / self.scaling_factor
        distances = distances.permute(0, 3, 1, 2)

        # Create an upper triangular mask for a single sequence
        single_upper_triangular_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

        # Expand the mask to cover all heads and batch size
        upper_triangular_mask = single_upper_triangular_mask.unsqueeze(0).unsqueeze(0)
        upper_triangular_mask = upper_triangular_mask.expand(batch_size, self.nhead, -1, -1).to(distances.device)

        # Apply the mask to distances
        distances = distances.masked_fill(upper_triangular_mask, float('-inf'))

        # Apply softmax to get attention scores
        attn = F.softmax(distances, dim=-1) #(bs, num_heads, seq_len, seq_len)
        mu_v = mu_v.permute(0, 2, 1, 3) # (bs, num_heads, seq_len, d_model//num_heads)
        # Perform matrix multiplication
        output_mu = torch.matmul(attn, mu_v) #(bs, num_heads, seq_len, d_model//num_heads)

        # # Optionally, reshape to concatenate results from all heads
        # output_mu = output_mu.reshape(batch_size, seq_len, -1)

        # Propagate uncertainties: Compute weighted sum of variances, then take square root
        var_v = sigma_v ** 2
        var_v = var_v.permute(0, 2, 1, 3)
        output_var = torch.clamp(torch.matmul(attn, var_v), min=0)
        output_sigma = torch.sqrt(output_var + epsilon)


        return output_mu, output_sigma


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_middle, dropout):
        super().__init__()
        self.l_feed = nn.Linear(d_model, d_middle)
        self.l_proj = nn.Linear(d_middle, d_model)
        self.dropout = nn.Dropout(dropout)
        self.NG = NewGELU()

    def forward(self, x):
        return self.dropout(self.l_proj(self.NG(self.l_feed(x))))


class ProbabilisticTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dim_out):
        super().__init__()
        self.attention = ProbabilisticAttention(d_model, num_heads)
        self.splitter_mu = nn.Linear(2 * d_model, d_model)
        self.splitter_sigma = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(2 * d_model)
        self.norm2 = nn.LayerNorm(2 * dim_out)
        self.dropout = nn.Dropout(0.1)
        self.probabilistic_token_transformer = ProbabilisticTokenTransformer(d_model, d_model)
        self.probabilistic_embedding = ProbabilisticEmbedding(2 * d_model, d_model)
        self.feedforward = FeedForward(2 * d_model, 4 * d_model, .1)

    def forward(self, x):
        # apply layer norm
        bs, seq_len, _ = x.size()
        normed_x = self.norm1(x)

        # Then we do linear layers to split x to mu and sigma
        mu = self.splitter_mu(normed_x)
        sigma = self.splitter_sigma(normed_x)

        # Transform tokens into means and standard deviations for Q, K, V
        (query, query_sigma), (key, key_sigma), (value, value_sigma) = self.probabilistic_token_transformer((mu, sigma))

        # Ensure query_sigma, key_sigma, and value_sigma are positive
        query_sigma = F.softplus(query_sigma)
        key_sigma = F.softplus(key_sigma)
        value_sigma = F.softplus(value_sigma)

        # Compute attention, separately handling means and standard deviations
        attn_output_mu, attn_output_sigma = self.attention(query, key, value, query_sigma, key_sigma, value_sigma)

        # Reshape and transpose to get [bs, seq_len, d_model]
        attn_output_mu = attn_output_mu.transpose(1, 2).reshape(bs, seq_len, -1)
        attn_output_sigma = attn_output_sigma.transpose(1, 2).reshape(bs, seq_len, -1)
        combined = torch.cat([attn_output_mu, attn_output_sigma], dim=-1)

        # apply dropout
        dropped = self.dropout(combined)

        x_attn = x + dropped  # skip connection
        normed_x_attn = self.norm1(x_attn)

        feed_x = self.feedforward(normed_x_attn)  # this does dropout

        output = normed_x_attn + feed_x  # skip connection

        return output


class ProbabilisticTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        vocab_size = kwargs['vocab_size']
        d_model = kwargs['d_model']
        num_heads = kwargs['num_heads']
        num_blocks = kwargs['num_blocks']
        self.embedding_in = nn.Embedding(vocab_size, 2 * d_model)
        self.probabilistic_embedding = ProbabilisticEmbedding(2 * d_model, d_model)
        self.transformer_blocks = nn.ModuleList([
            ProbabilisticTransformerBlock(d_model, num_heads, 2 * d_model, d_model)
            for _ in range(num_blocks)
        ])
        self.output_projection = nn.Linear(2 * d_model, vocab_size, bias=False)
        self.out_mu = nn.Linear(2 * d_model, d_model, bias=False)
        self.out_sigma = nn.Linear(2 * d_model, d_model, bias=False)
        self.positional_encoding = PositionalEncoding(2 * d_model)

    def forward(self, input_ids):
        embedding_in = self.embedding_in(input_ids)
        x = self.positional_encoding(embedding_in)
        for block in self.transformer_blocks:
            x = block(x)
        logits = self.output_projection(x)
        mu_out = self.out_mu(x)
        sigma_out = F.softplus(self.out_sigma(x))
        return logits, mu_out, sigma_out

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
