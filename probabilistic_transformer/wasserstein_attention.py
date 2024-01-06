import torch
import torch.nn as nn
import torch.nn.functional as F

class WassersteinDistance(nn.Module):
    def forward(self, mu1, mu2, sigma1, sigma2):
        term1 = torch.sum((mu1 - mu2) ** 2, dim=-1)
        term2 = torch.sum(sigma1 ** 2 + sigma2 ** 2 - 2 * (sigma1 * sigma2), dim=-1)
        W_distance = torch.sqrt(term1 + term2)
        return W_distance

class MultiHeadWassersteinAttention(nn.Module):
    def __init__(self, d_model, num_heads, alpha=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.wasserstein_distance = WassersteinDistance()
        self.linear = nn.Linear(d_model, d_model)  # Project back to original d_model
        # Separate linear layers for mu
        self.mu_query = nn.Linear(d_model, d_model)
        self.mu_key = nn.Linear(d_model, d_model)
        self.mu_value = nn.Linear(d_model, d_model)
        self.combined_qkv_mu = nn.Linear(d_model, 3 * d_model)
        self.combined_qkv_sigma = nn.Linear(d_model, 3 * d_model)
        # Separate linear layers for sigma
        self.sigma_query = nn.Linear(d_model, d_model)
        self.sigma_key = nn.Linear(d_model, d_model)
        self.sigma_value = nn.Linear(d_model, d_model)

        self.linear = nn.Linear(d_model, d_model)  # Project back to original d_model
        self.alpha = alpha

    def forward(self, mu, sigma):
        bs, L, d_model = mu.size()  # batch size, sequence length, d_model
        d_k = d_model // self.num_heads

        # Separate linear projections for mu and sigma
        Q_mu, K_mu, V_mu = self.combined_qkv_mu(mu).split(d_model, dim=2)
        Q_sigma, K_sigma, V_sigma = self.combined_qkv_sigma(sigma).split(d_model, dim=2)

        # Reshape and transpose for multi-head attention
        def reshape_transpose(x):
            return x.view(bs, L, self.num_heads, d_k).transpose(1, 2)

        Q_mu, K_mu, V_mu = map(reshape_transpose, (Q_mu, K_mu, V_mu))
        Q_sigma, K_sigma, V_sigma = map(reshape_transpose, (Q_sigma, K_sigma, V_sigma))

        # Compute pairwise Wasserstein distance for all heads in parallel
        W_distance = self.wasserstein_distance(Q_mu.unsqueeze(3), K_mu.unsqueeze(2), Q_sigma.unsqueeze(3), K_sigma.unsqueeze(2))
        similarity = torch.exp(-self.alpha * W_distance)
        attention_weights = F.softmax(similarity, dim=-1)

        # Apply attention weights to mu_V and sigma_V
        weighted_mu_V = torch.einsum('bnij,bnjk->bnik', attention_weights, V_mu)
        weighted_sigma_V = torch.einsum('bnij,bnjk->bnik', attention_weights, V_sigma)

        # Combine heads and project back to original feature dimension for both mu and sigma
        weighted_mu_V = weighted_mu_V.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        weighted_sigma_V = weighted_sigma_V.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)

        mu_attn = self.linear(weighted_mu_V)
        sigma_attn = self.linear(weighted_sigma_V)

        return mu_attn, sigma_attn


    # def forward(self, mu, sigma):
    #     bs, seq_len, _ = mu.size()
    #
    #     # Reshape mu and sigma to [bs, num_heads, seq_len, head_dim]
    #     mu = mu.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    #     sigma = sigma.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    #
    #     # Compute pairwise Wasserstein distance for all heads in parallel
    #     mu_expanded = mu.unsqueeze(3)
    #     sigma_expanded = sigma.unsqueeze(3)
    #     mu_t = mu.unsqueeze(2)
    #     sigma_t = sigma.unsqueeze(2)
    #
    #     W_distance = self.wasserstein_distance(mu_expanded, mu_t, sigma_expanded, sigma_t)
    #     similarity = torch.exp(-self.alpha * W_distance)
    #     attention_weights = F.softmax(similarity, dim=-1)
    #
    #     # Apply attention weights to mu (assuming mu are the value vectors)
    #     weighted_mu = torch.einsum('bnij,bnjk->bnik', attention_weights, mu)
    #
    #     # Combine heads and project back to original feature dimension
    #     weighted_mu = weighted_mu.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
    #     multihead_output = self.linear(weighted_mu)
    #
    #     return multihead_output

# Example usage
d_model = 4  # Size of the feature dimension for each head
num_heads = 2  # Number of attention heads
  # Batch size, sequence length, and number of features
bs = 2
seq_len = 5
mu = torch.randn(bs, seq_len, d_model)  # Example means
sigma = torch.rand(bs, seq_len, d_model)  # Example standard deviations (positive)

# Initialize the multi-headed attention module
multihead_wasserstein_attention = MultiHeadWassersteinAttention(d_model, num_heads)

# Calculate the multi-headed attention output
multihead_attention_output = multihead_wasserstein_attention(mu, sigma)

print(multihead_attention_output)










