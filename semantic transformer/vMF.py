import torch
import torch.nn as nn
import torch.nn.functional as F

class vMFNet(nn.Module):
    def __init__(self, **kwargs):
        self.input_dim = kwargs.get('input_dim')
        self.hidden_dim = kwargs.get('hidden_dim')
        self.output_dim = kwargs.get('output_dim')
        self.n_components = kwargs.get('n_components')
        self.dropout_p = kwargs.get('dropout_p')
        super(vMFNet, self).__init__()

        # Define the layers
        self.hidden1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden2 = nn.Linear(self.hidden_dim, 2*self.hidden_dim)
        self.hidden3 = nn.Linear(2*self.hidden_dim, 2*self.hidden_dim)
        self.hidden4 = nn.Linear(2*self.hidden_dim, self.hidden_dim)

        self.mu_output = nn.Linear(self.hidden_dim, self.n_components * self.output_dim)
        self.weights_output = nn.Linear(self.hidden_dim, self.n_components)

        # Define dropout
        self.dropout = nn.Dropout(self.dropout_p)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        current_batch_size, seq_len, output_dim = x.size()
        x = x.view(current_batch_size * seq_len, output_dim)
        # Forward propagate through the layers
        x = F.gelu(self.dropout(self.hidden1(x)))
        x = F.gelu(self.dropout(self.hidden2(x)))
        x = F.relu(self.dropout(self.hidden3(x)))
        x = F.relu(self.dropout(self.hidden4(x)))

        mu_outputs = self.mu_output(x)
        weight_outputs = self.weights_output(x)

        # Compute the mean directions and mixture weights
        mus = F.normalize(mu_outputs.reshape(-1, self.n_components, self.output_dim), dim=-1)
        weights = self.softmax(weight_outputs)

        return (mus, weights)


def loss(params, targets):
    mus, weights = params
    log_likelihoods = (mus * targets.unsqueeze(1)).sum(-1)
    weighted_log_likelihoods = weights * log_likelihoods
    return -weighted_log_likelihoods.sum(-1).mean()


def sample_from_mixture(mus, weights):
    # Shape of mus: (batch_size, n_components, d_dimensions)
    # Shape of weights: (batch_size, n_components)

    # Choose a component for each data point based on the weights
    component_indices = torch.multinomial(weights, num_samples=1).squeeze()

    # Use advanced indexing to select the right component for each data point
    samples = mus[torch.arange(mus.size(0)), component_indices]

    return samples