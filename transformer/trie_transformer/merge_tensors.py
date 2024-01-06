import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        # Assuming the feature dimensions of t1 and t2 are the same
        self.query_linear = nn.Linear(feature_dim, feature_dim)
        self.key_linear = nn.Linear(feature_dim, feature_dim)
        self.value_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, t1, t2):
        # Generate queries, keys, and values
        queries = self.query_linear(t1)
        keys = self.key_linear(t2)
        values = self.value_linear(t2)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))

        # Apply mask (add a large negative value to masked positions)
        mask = torch.tril(torch.ones(t1.size(1), t2.size(1))).to(t1.device)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Normalize scores
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute weighted sum of values
        attention_output = torch.matmul(attention_weights, values)
        return attention_output

# Example usage
feature_dim = 3  # Example feature dimension
model = CrossAttention(feature_dim=feature_dim)

# Example tensors
t1 = torch.randn(3, feature_dim)  # Tensor 1
t2 = torch.randn(3, feature_dim)  # Tensor 2

# Apply the model
result = model(t1, t2)
print(result)


