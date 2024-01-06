import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np



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


class ProbabilisticEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_mu = nn.Linear(vocab_size, embedding_dim)
        self.embedding_sigma = nn.Linear(vocab_size, embedding_dim)

    def forward(self, input_ids):
        mu = self.embedding_mu(input_ids)
        sigma = F.softplus(self.embedding_sigma(input_ids))  # Ensure sigma is positive
        return mu, sigma


class ProbabilisticMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=.1):
        super().__init__()
        self.combined_qkv_mu = nn.Linear(d_model, 3 * d_model)
        self.combined_qkv_sigma = nn.Linear(d_model, 3 * d_model)
        self.linear_proj_mu = nn.Linear(num_heads * (d_model // num_heads), d_model)
        self.linear_proj_sigma = nn.Linear(num_heads * (d_model // num_heads), d_model)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.alpha = nn.Parameter(torch.tensor(.1))

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

        # Attention scores with uncertainty
        scores_mu = Q_mu @ K_mu.transpose(-2, -1) / (d_k ** 0.5)
        scores_sigma = Q_sigma @ K_sigma.transpose(-2, -1) / (d_k ** 0.5)

        # Combine uncertainty
        scores = scores_mu + self.alpha * scores_sigma

        # Mask subsequent positions and apply softmax
        scores = scores + torch.triu(torch.ones_like(scores) * float("-inf"), diagonal=1)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to value vectors
        attn_mu = attn @ V_mu
        attn_sigma = attn @ V_sigma

        # Concatenate heads and apply linear projection
        attn_mu = attn_mu.transpose(1, 2).contiguous().view(bs, L, d_model)
        attn_sigma = attn_sigma.transpose(1, 2).contiguous().view(bs, L, d_model)

        attn_mu = self.dropout(self.linear_proj_mu(attn_mu))
        attn_sigma = self.dropout(self.linear_proj_sigma(attn_sigma))

        return attn_mu, attn_sigma


class ProbabilisticTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attention = MultiHeadWassersteinAttention(d_model,
                                                                 num_heads)  # ProbabilisticMultiHeadAttention(d_model, num_heads)
        self.norm1_mu = nn.LayerNorm(d_model)
        self.softplus = nn.Softplus()
        self.norm1_sigma = nn.LayerNorm(d_model)
        self.norm2_mu = nn.LayerNorm(d_model)
        self.norm2_sigma = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.feedforward_mu = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            self.dropout,
            nn.Linear(4 * d_model, d_model)
        )
        self.feedforward_sigma = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            self.dropout,
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, mu, sigma):
        # Multi-head attention

        normed_mu = self.norm1_mu(mu)
        # normed_sigma = self.norm1_sigma(sigma)
        soft_sigma = self.softplus(sigma)
        attn_mu, attn_sigma = self.multihead_attention(normed_mu, soft_sigma)
        # attn_mu, attn_sigma = self.multihead_attention(normed_mu, normed_sigma)
        mu = mu + attn_mu
        sigma = sigma + attn_sigma
        normed_mu = self.norm2_mu(mu)
        soft_sigma = self.softplus(sigma)
        # normed_sigma = self.norm2_sigma(sigma)
        mu = mu + self.feedforward_mu(normed_mu)
        sigma = sigma + self.feedforward_sigma(soft_sigma)  # self.feedforward_sigma(normed_sigma)

        return mu, sigma


class ProbabilisticOutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim, training=True):
        super().__init__()
        self.output_mu = nn.Linear(input_dim, output_dim)
        self.output_sigma = nn.Linear(input_dim, output_dim, bias=False)
        self.training = training

    def forward(self, mu, sigma):
        logits_mu = self.output_mu(mu)
        epsilon = 1e-6
        logits_sigma = F.softplus(self.output_sigma(sigma)) + epsilon  # to ensure sigma is positive

        # During training, sample from the distribution; during evaluation, use mu directly
        if self.training:
            eps = torch.randn_like(logits_sigma)
            sampled_logits = logits_mu + logits_sigma * eps
        else:
            sampled_logits = logits_mu

        return sampled_logits, logits_sigma


class ProbabilisticOutputLayer2(nn.Module):
    def __init__(self, input_dim, output_dim, training=True):
        super().__init__()
        self.output_mu1 = nn.Linear(input_dim, input_dim, bias=False)
        self.output_sigma1 = nn.Linear(input_dim, input_dim, bias=False)
        self.combined1 = nn.Linear(2 * input_dim, 2 * input_dim, bias=False)
        self.combined2 = nn.Linear(2 * input_dim, 2 * input_dim, bias=False)
        self.split_mu = nn.Linear(2 * input_dim, input_dim, bias=False)
        self.split_sigma = nn.Linear(2 * input_dim, input_dim, bias=False)
        self.out_mu = nn.Linear(input_dim, output_dim, bias=False)
        self.out_sigma = nn.Linear(input_dim, output_dim, bias=False)

        self.training = training

    def forward(self, mu, sigma):
        mu1 = self.output_mu1(mu)
        mu1 = F.relu(mu1)
        sigma1 = self.output_sigma1(sigma)
        sigma1 = F.relu(sigma1)
        comb = torch.cat([mu1, sigma1], dim=-1)
        combined1 = self.combined1(comb)
        combined1 = F.relu(combined1)
        combined2 = self.combined2(combined1)
        combined2 = F.relu(combined2)
        mu_split = self.split_mu(combined2)
        mu_split = F.relu(mu_split)
        sigma_split = self.split_sigma(combined2)
        sigma_split = F.relu(sigma_split)
        out_mu = self.out_mu(mu_split)
        out_sigma = self.out_sigma(sigma_split)

        epsilon = 1e-6
        logits_sigma = F.softplus(out_sigma) + epsilon  # to ensure sigma is positive

        # During training, sample from the distribution; during evaluation, use mu directly
        if self.training:
            eps = torch.randn_like(logits_sigma)
            sampled_logits = out_mu + logits_sigma * eps
        else:
            sampled_logits = out_mu

        return sampled_logits, logits_sigma


class ProbabilisticTransformer(nn.Module):
    def __init__(self, training=True, **kwargs):
        super().__init__()
        vocab_size = kwargs['vocab_size']
        d_model = kwargs['d_model']
        num_heads = kwargs['num_heads']
        num_blocks = kwargs['num_blocks']
        self.embedding_in = nn.Embedding(vocab_size, 2 * d_model)
        self.probabilistic_embedding = ProbabilisticEmbedding(2 * d_model, d_model)
        self.transformer_blocks = nn.ModuleList([
            ProbabilisticTransformerBlock(d_model, num_heads)
            for _ in range(num_blocks)
        ])
        self.output_projection = ProbabilisticOutputLayer(d_model, vocab_size,
                                                          training)  # nn.Linear(2*d_model, vocab_size, bias=False)
        self.positional_encoding = PositionalEncoding(2*d_model)

    def forward(self, input_ids):
        embedding_in = self.embedding_in(input_ids)
        pos = self.positional_encoding(embedding_in)
        embedding_in += pos
        mu, sigma = self.probabilistic_embedding(embedding_in)

        for block in self.transformer_blocks:
            mu, sigma = block(mu, sigma)

        # combined = torch.cat([mu, sigma], dim=2)
        logits, sigma = self.output_projection(mu, sigma)  # self.output_projection_mu(combined)
        return logits, sigma
    def configure_optimizers(self, model_params, lr_params):
        return None

# create a closure for train_output so it can store the tokenizer
def create_train_output(tokenizer, loss_func):
    def train_output(**kwargs):
        pred = kwargs['pred']
        targets = kwargs['target']
        log_data = kwargs['log_data']
        loss_func(pred, targets, debug=True)

    return train_output


# def classification_and_sigma(pred, targets, epsilon=1e-6):
#     mu_pred, sigma_pred = pred
#     bs, seq_len, vocab_size = mu_pred.shape
#     # Ensure sigma is not too small to avoid division by a very small number
#     sigma_pred_squared_clipped = torch.max(sigma_pred**2, torch.tensor(epsilon).to(sigma_pred.device))
#     sigma_pred_squared_clipped = sigma_pred_squared_clipped.view(-1, vocab_size)
#     mu_pred = mu_pred.view(-1, vocab_size)
#     targets = targets.view(-1)
#     prediction_loss = F.cross_entropy(mu_pred, targets)
#
#     # Uncertainty loss
#     true_class_probs = F.softmax(mu_pred, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
#     true_class_sigmas = sigma_pred_squared_clipped.gather(1, targets.unsqueeze(1)).squeeze(1)
#     true_class_errors = (1 - true_class_probs) ** 2
#     uncertainty_loss = torch.mean(true_class_errors / true_class_sigmas + torch.log(true_class_sigmas))
#
#     # Combine the losses
#     total_loss = prediction_loss + .1 * torch.log(uncertainty_loss)
#     return total_loss

#
# def classification_and_sigma(pred, targets, epsilon=1e-8, debug=False):
#     mu_pred, raw_sigma_pred = pred
#     bs, seq_len, vocab_size = mu_pred.shape
#
#     sigma_pred_squared = raw_sigma_pred**2 + epsilon
#     # Ensure sigma squared is not too small
#
#     # Reshape for cross entropy
#     mu_pred = mu_pred.view(-1, vocab_size)
#     targets = targets.view(-1)
#
#     # Prediction loss
#     prediction_loss = F.cross_entropy(mu_pred, targets)
#
#     # Softmax for calculating true class probabilities
#     true_class_probs = F.softmax(mu_pred, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
#
#     # Gather the corresponding sigmas for the true classes and clip them
#     true_class_sigmas = sigma_pred_squared.view(-1, vocab_size).gather(1, targets.unsqueeze(1)).squeeze(1)
#
#     # Calculate errors for the true class probabilities
#     true_class_errors = .1*(1 - true_class_probs) ** 2
#
#     # Uncertainty loss component
#     uncertainty_component = torch.mean(true_class_errors / true_class_sigmas) + 10 * true_class_sigmas.mean()
#
#     # Regularization term for sigma, to encourage smaller sigmas
#     sigma_regularization = torch.mean(sigma_pred_squared)
#
#     # Combine uncertainty component and regularization, ensuring no negative values
#     uncertainty_loss = uncertainty_component + sigma_regularization
#
#     # Combine the losses, scale the uncertainty loss
#     total_loss = prediction_loss + .1 * uncertainty_loss
#
#     if debug:
#         print(f'true_class_probs {true_class_probs.mean()}')
#         print(f'true_class_errors {true_class_errors.mean()}')
#         print(f'true_class_sigmas', {true_class_sigmas.mean()})
#         print(f'sigma_regularization', sigma_regularization)
#         print(f'Uncertainty_component {uncertainty_component.mean()}')
#
#     return total_loss
#
# def certainty_penalty(sigma, beta, epsilon=0):
#     # Apply only the logarithmic penalty
#     log_penalty = -beta * torch.log(sigma + epsilon)
#     return log_penalty
#
# true_class_error_min = torch.tensor(1.)
# def classification_and_sigma(pred, targets, alpha=0.1, epsilon=1e-8, debug=False):
#     global true_class_error_min
#     mu_pred, raw_sigma_pred = pred
#     bs, seq_len, vocab_size = mu_pred.shape
#     # Ensure sigma squared is not too small
#     sigma_pred_squared = raw_sigma_pred**2 + epsilon
#
#     # Reshape for cross entropy
#     mu_pred = mu_pred.view(-1, vocab_size)
#     targets = targets.view(-1)
#
#     # Prediction loss
#     prediction_loss = F.cross_entropy(mu_pred, targets)
#
#     # Softmax for calculating true class probabilities
#     true_class_probs = F.softmax(mu_pred, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
#
#     # Gather the corresponding sigmas for the true classes and clip them
#     true_class_sigmas = torch.sqrt(sigma_pred_squared.view(-1, vocab_size).gather(1, targets.unsqueeze(1)).squeeze(1))
#     max_prob_scalar = true_class_probs.max().item()
#     min_error_scalar = (1 - max_prob_scalar) ** 2
#
#     # Calculate errors for the true class probabilities
#     true_class_errors = (1 - true_class_probs) ** 2
#     m = true_class_errors.mean()
#     # if m < true_class_error_min:
#     #     true_class_error_min = m
#     # Calculate the uncertainty loss using the custom loss function
#
#     uncertainty_loss = custom_loss_torch(true_class_errors, true_class_sigmas, m, a=0)
#     uncertainty_loss = torch.mean(uncertainty_loss)  # Take the mean to get a single loss value
#
#     # Regularization term for sigma, to encourage smaller sigmas
#     sigma_regularization = torch.mean(sigma_pred_squared)
#     certainty_loss = certainty_penalty(sigma_regularization,1)
#     # Combine the losses, scale the uncertainty loss
#     a = 1
#     b = 1
#     c = .01
#     total_loss = 10 * prediction_loss + a*uncertainty_loss + b* sigma_regularization #+ c * certainty_loss
#
#     if debug:
#         print(f'total_loss {total_loss}')
#         print(f'classification loss {prediction_loss}')
#         print(f'Uncertainty_component {uncertainty_loss}, scaled: {a*uncertainty_loss}')
#         print(f'sigma_regularization {sigma_regularization}')
#         print(f'certainty_penalty {certainty_loss} scaled: {c* certainty_loss}')
#         print(f'true_class_probs {true_class_probs.mean()}')
#         print(f'true_class_errors {true_class_errors.mean()}')
#         print(f'true_class_sigmas {true_class_sigmas.mean()}')
#         print(f'batch error min {m}')
#     return total_loss
#
#
# def custom_loss_torch(error, sigma, mean_batch_error, a):
#     # Scale error and sigma
#     shifted_mean = mean_batch_error - .05
#     shifted_y = 10 * (error - shifted_mean)
#     b = torch.tensor(1)
#     s = torch.tensor(1)
#     e = torch.tensor(1)
#
#     # Apply the condition to each element of the error and sigma tensors
#     condition = .1 * error >= sigma
#     loss = torch.where(condition,
#                        1/s*(0.5*s-b)*sigma + 1/e*(b-a)*shifted_y + a,
#                        a + (a * sigma / (2 * s)) + (sigma / 2))
#     return loss
#
# import torch.nn.functional as F

def classification_and_sigma(pred, targets, alpha=0.1, epsilon=1e-8, debug=False):
    logits, sigmas = pred
    epsilon = 1e-6
    confidences = 1 / (sigmas + epsilon)

    # Reshape logits and confidences to combine batch and sequence_length dimensions
    batch_size, seq_len, vocab_size = logits.size()
    logits = logits.view(-1, vocab_size)
    confidences = confidences.view(-1, vocab_size)
    targets = targets.view(-1)

    # Scale logits by confidence
    scaled_logits = logits * confidences
    cross_entropy = F.cross_entropy(logits, targets)
    # Calculate cross-entropy loss with the scaled logits
    scaled_cross_entropy = F.cross_entropy(scaled_logits, targets, reduction='mean')

    # Add regularization term to encourage higher confidence
    reg_term = torch.mean(sigmas)
    lambda_reg = 1  # This is a hyperparameter that you'd need to tune
    loss = cross_entropy + .1 * (scaled_cross_entropy + lambda_reg * reg_term)

    if debug:
        print(f'cross_entropy {cross_entropy}')
        print(f'scaled_cross_entropy {scaled_cross_entropy}')
        print(f'confidences.mean() {confidences.mean()}')
        print(f'reg_term {reg_term}')
    return loss


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
        W_distance = self.wasserstein_distance(Q_mu.unsqueeze(3), K_mu.unsqueeze(2), Q_sigma.unsqueeze(3),
                                               K_sigma.unsqueeze(2))
        similarity = torch.exp(-self.alpha * W_distance)

        # Apply masking to the similarity scores
        mask = torch.triu(torch.ones_like(similarity) * float("-inf"), diagonal=1)
        masked_similarity = similarity + mask

        # Apply softmax to get attention weights
        attention_weights = F.softmax(masked_similarity, dim=-1)

        # Apply attention weights to mu_V and sigma_V
        weighted_mu_V = torch.einsum('bnij,bnjk->bnik', attention_weights, V_mu)
        weighted_sigma_V = torch.einsum('bnij,bnjk->bnik', attention_weights, V_sigma)

        # Combine heads and project back to original feature dimension for both mu and sigma
        weighted_mu_V = weighted_mu_V.transpose(1, 2).contiguous().view(bs, L, self.d_model)
        weighted_sigma_V = weighted_sigma_V.transpose(1, 2).contiguous().view(bs, L, self.d_model)

        mu_attn = self.linear(weighted_mu_V)
        sigma_attn = self.linear(weighted_sigma_V)

        return mu_attn, sigma_attn
