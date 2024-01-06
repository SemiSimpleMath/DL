
# def sigma_penalty_loss(logits, sigma, target, lambda_reg=0.01, target_sigma=1e-5):
#     # Calculate the base loss (e.g., cross-entropy) using the sampled logits
#     loss_f = nn.CrossEntropyLoss()
#     classification_loss = loss_f(logits, target)
#
#     # Calculate the penalty factor for sigma
#     min_penalty = 1.0
#     max_penalty = 2.0
#     # Ensure sigma is positive to avoid division by zero or negative values
#     sigma = torch.clamp(sigma, min=1e-6)
#     penalty_factor = torch.where(sigma > target_sigma,
#                                  torch.full_like(sigma, min_penalty),
#                                  min_penalty + (max_penalty - min_penalty) * (target_sigma - sigma) / target_sigma)
#
#     # Apply confidence penalty to the loss
#     weighted_loss = classification_loss * penalty_factor
#
#     # Uncertainty regularization term to prevent the model from being too uncertain
#     uncertainty_penalty = lambda_reg * torch.mean(sigma)
#
#     # Combine the weighted classification loss with the uncertainty penalty
#     total_loss = torch.mean(weighted_loss) + uncertainty_penalty
#
#     return total_loss

import torch.nn as nn
def sigma_penalty_loss(logits, sigma, target, epsilon=1e-6):

    loss_f = nn.CrossEntropyLoss()
    cel = loss_f(logits, target)
    total_losses = []
    sigma_threshold = 1e-5
    scaled_cel = sigma_threshold * cel
    sigma = sigma.mean()
    # Apply the conditional logic based on the sigma value
    if sigma < sigma_threshold:
        # Calculate the loss for this sigma value based on the condition
        loss = (scaled_cel/(sigma+epsilon) + cel)/2
    else:
        # Calculate the loss for this sigma value with the default formula
        loss = cel + sigma

import torch
import torch.nn.functional as F

def custom_loss(targets, mu_pred, sigma_pred, epsilon=1e-8):
    # compute loss
    mu_pred = mu_pred.permute(0, 2, 1)
    # Ensure sigma is not too small to avoid division by a very small number
    sigma_pred_squared_clipped = torch.max(sigma_pred**2, torch.tensor(epsilon).to(sigma_pred.device))

    # Flatten mu_pred for cross entropy, keeping the class dimension

    # Flatten mu_pred for cross entropy, keeping the class dimension TODO Sort this out!
    prediction_loss = F.cross_entropy(mu_pred.view(-1, num_classes), targets.view(-1), reduction='mean')

    # Prediction loss
    prediction_loss = F.cross_entropy(mu_pred.view(-1, mu_pred.size(-1)), targets.view(-1), reduction='none')
    prediction_loss = prediction_loss.view(targets.size()).mean()

    # Uncertainty loss
    true_class_probs = F.softmax(mu_pred, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
    true_class_sigmas = sigma_pred_squared_clipped.gather(1, targets.unsqueeze(1)).squeeze(1)
    true_class_errors = (1 - true_class_probs) ** 2
    uncertainty_loss = torch.mean(true_class_errors / true_class_sigmas + torch.log(true_class_sigmas))


    # Calculate the squared error for the true class prediction
    true_class_errors = (1 - true_class_probs) ** 2

    # The uncertainty loss for the true class
    uncertainty_loss = torch.mean(true_class_errors / true_class_sigmas + torch.log(true_class_sigmas))

    # Combine the losses
    total_loss = prediction_loss + uncertainty_loss
    return total_loss





# Redefine the loss function with the corrected shape for sigma
def loss_with_uncertainty_attenuation_torch_corrected(y_true, p_pred, sigma, lambda_param=0.5):
    # Calculate the cross-entropy loss for each prediction
    ce_loss = torch_cross_entropy_loss(y_true, p_pred)

    # Select the sigma values for the predicted classes
    _, predicted_classes = torch.max(p_pred, -1)
    sigma_selected = torch.gather(sigma, -1, predicted_classes.unsqueeze(-1)).squeeze(-1)

    # Calculate the uncertainty attenuation
    # We add a small constant to sigma_selected to avoid division by zero
    uncertainty_attenuation = ce_loss / (sigma_selected + 1e-6)

    # Calculate the uncertainty penalty
    uncertainty_penalty = torch.sum(sigma_selected ** 2)

    # Combine the loss components
    total_loss = torch.sum(uncertainty_attenuation) + lambda_param * uncertainty_penalty

    return total_loss / y_true.numel()  # Normalize by the total number of items in the batch

    # Example data in PyTorch tensors
    # For simplicity, let's assume bs=2, seq_len=3, vocab=4
    bs, seq_len, vocab = 2, 3, 4
    y_true = torch.tensor([[0, 2, 1], [1, 3, 0]])  # Shape: [bs, seq_len]
    p_pred = torch.randn(bs, seq_len, vocab)  # Shape: [bs, seq_len, vocab]
    sigma = torch.rand(bs, seq_len, vocab)  # Shape: [bs, seq_len, vocab]

    # Calculate the loss with PyTorch tensors
    total_loss_pytorch_corrected = loss_with_uncertainty_attenuation_torch_corrected(y_true, p_pred, sigma)
    total_loss_pytorch_corrected.item()


def loss_with_uncertainty_attenuation(p_pred, sigma, target, lambda_param=0.5):
    # Cross-entropy loss component
    criterion = nn.CrossEntropyLoss()
    ce_loss = criterion(p_pred, target)

    # Uncertainty attenuation component
    # Here we use sigma as a weight for the cross-entropy loss, assuming higher sigma means lower confidence
    # We add a small constant to the denominator to avoid division by zero
    uncertainty_attenuation = np.sum(ce_loss / (sigma + 1e-6))

    # Uncertainty regularization component
    # This is the penalty for high uncertainty (high sigma)
    uncertainty_penalty = np.sum(sigma ** 2)

    # Combine the components to get the total loss
    total_loss = uncertainty_attenuation + lambda_param * uncertainty_penalty
    return total_loss / len(target)


def calculate_penalty(confidence, surprise, beta=1.0, gamma=0.5):
    high_conf_high_surp_penalty = confidence * surprise * beta
    low_surprise_low_confidence = (1 - confidence) * (1 - surprise) * gamma
    total_penalty = high_conf_high_surp_penalty + low_surprise_low_confidence
    return total_penalty



def calculate_surprise(logits, targets):
    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Expand targets to match the shape of probabilities
    targets_expanded = targets.unsqueeze(-1).expand_as(probabilities)

    # Create a range tensor that matches the last dimension of probabilities
    range_tensor = torch.arange(probabilities.size(-1), device=logits.device).unsqueeze(0).unsqueeze(0).expand_as(probabilities)

    # Calculate surprise: 1 - P for target class, P for non-target classes
    surprise = torch.where(range_tensor == targets_expanded, 1 - probabilities, probabilities)

    return surprise

def surprise_loss(preds, targets, beta=1.0, gamma=0.5):
    # Calculate base loss (cross-entropy)
    logits, sigmas = preds
    base_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')

    # Calculate confidence (inverse of sigma)
    confidence = 1 - F.softplus(sigmas)
    # Calculate surprise
    surprise = calculate_surprise(logits, targets)
    # Calculate penalty
    penalty = calculate_penalty(confidence, surprise, beta, gamma)

    # Combine base loss and penalty
    total_loss = base_loss + penalty.mean() + sigmas.mean()
    return total_loss

#
# import torch
# import torch.nn.functional as F
#
# # Example logits from a neural network's output layer
# logits = torch.tensor([...])  # Replace with your actual logits
#
# # Apply softmax to convert logits to probabilities
# softmax_probs = F.softmax(logits, dim=0)
#
# # The index of the target class that was observed
# target_class_index = ...
#
# # The likelihood of observing each class. 1 for the target class, 0 for others
# likelihoods = torch.zeros_like(softmax_probs)
# likelihoods[target_class_index] = 1
#
# # Calculate the evidence (since likelihood is 0 for all non-target classes, this simplifies)
# evidence = softmax_probs[target_class_index]
#
# # Update the probabilities for all classes
# new_priors = (likelihoods * softmax_probs) / evidence
#
# softmax_probs, new_priors
