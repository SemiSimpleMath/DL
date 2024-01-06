import matplotlib.pyplot as plt
import numpy as np

def calculate_losses(cel, sigma_range, epsilon=1e-6):
    # Iterate over each sigma value in the sigma_range
    total_losses = []
    sigma_threshold = 1e-4
    scaled_cel = sigma_threshold * cel
    for sigma in sigma_range:
        # Apply the conditional logic based on the sigma value
        if sigma < sigma_threshold:
            # Calculate the loss for this sigma value based on the condition
            loss = (scaled_cel/(sigma+epsilon) + cel)/2
        else:
            # Calculate the loss for this sigma value with the default formula
            loss = cel + sigma

        # Append the calculated loss to the total_losses list
        total_losses.append(loss)

    # Convert the list of losses to a NumPy array if needed
    total_losses = np.array(total_losses)
    return total_losses

# # Define the sigma and beta ranges
# sigma_range = np.linspace(1e-5, 1e-3, 100)  # Avoid zero
# beta_range = np.logspace(-3, -1, num=3, base=10.0)
#
# # Define a range of CEL values
# cel_range = [1, 3, 5, 7, 10]
#
# # Plot for each CEL value
# for cel in cel_range:
#     plt.figure()  # Create a new figure for each CEL value
#     for beta in beta_range:
#         losses = calculate_losses(cel, sigma_range, beta)
#         plt.plot(sigma_range, losses, label=f'Beta = {beta:.1e}, CEL = {cel}')
#
#     plt.title(f'Total Loss vs. Sigma (CEL = {cel})')
#     plt.xlabel('Sigma')
#     plt.ylabel('Total Loss')
#     #plt.yscale('log')
#     #plt.xscale('log')
#     plt.legend()
#     plt.show()
#
# import numpy as np

# Define the cross-entropy loss function
def cross_entropy_loss(y_true, p_pred):
    # Avoid division by zero
    p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)
    # Compute the cross-entropy loss
    return -np.sum(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))
# Redefine the confidence penalty function with the new interpretation of sigma
def confidence_penalty_new_interpretation(y_true, p_pred, sigma):
    # Determine where the predictions are correct (1 for correct, 0 for incorrect)
    correct_predictions = (y_true == (p_pred > 0.5)).astype(float)
    # Define the confidence loss with the new interpretation of sigma
    conf_loss = (sigma ** 2 * correct_predictions) + ((1 - sigma) ** 2 * (1 - correct_predictions))
    return np.sum(conf_loss)

# Define the combined loss function with the new confidence penalty
def combined_loss_new_interpretation(y_true, p_pred, sigma, lambda_param=0.5):
    ce_loss = cross_entropy_loss(y_true, p_pred)
    conf_loss = confidence_penalty_new_interpretation(y_true, p_pred, sigma)
    # Combine the losses
    total_loss = ce_loss + lambda_param * conf_loss
    return total_loss




# Example data
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
p_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.9, 0.1, 0.6, 0.2])
sigma = np.array([0.8, 0.7, 0.9, 0.6, 0.5, 0.4, 0.9, 0.6, 0.7, 0.3])

# Calculate the combined loss
# Calculate the combined loss with the new interpretation of sigma
total_loss = combined_loss_new_interpretation(y_true, p_pred, sigma)
print(total_loss)
