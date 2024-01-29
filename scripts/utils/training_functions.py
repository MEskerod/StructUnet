import torch

#### LOSS FUNCTIONS AND ERROR METRICS ####
def dice_loss(inputs, targets, smooth=1e-5):
  intersection = torch.sum(targets * inputs, dim=(1,2,3))
  sum_of_squares_pred = torch.sum(torch.square(inputs), dim=(1,2,3))
  sum_of_squares_true = torch.sum(torch.square(targets), dim=(1,2,3))
  dice = (2 * intersection + smooth) / (sum_of_squares_pred + sum_of_squares_true + smooth)
  return 1-dice

def mse_loss(inputs, targets):
    return ((inputs - targets) ** 2).mean()

# Asymmetric imposes larger penalty on false negatives than false positives
#TODO - Update what is positive and negative
def f1_score(inputs, targets, epsilon=1e-7, treshold = 0.5):
    # Ensure tensors have the same shape
    assert inputs.shape == targets.shape

    binary_input = (inputs >= treshold).float()

    # Calculate true positives, false positives, and false negatives
    #Black (0) is considered the positive
    true_positives = torch.sum((1 - targets) * (1 - binary_input))
    false_positives = torch.sum(targets * (1 - binary_input))
    false_negatives = torch.sum((1 - inputs) * binary_input)

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1

#### OPTIMIZER ####
def adam_optimizer(model, lr):
  return torch.optim.Adam(model.parameters(), lr=lr)

#TODO - Look at maybe other optimizers, but at least weight decay!!!



#### PLOTS DURING TRAINING ####


#### FUNCTION FOR TRANING ####


