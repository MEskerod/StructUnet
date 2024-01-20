from python_packages import *

##### SETTING UP DATA ####
def move_files(src_dir, dst_dir, file_list):
    """
    """
    for filename in file_list:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        os.rename(src_path, dst_path)

def split_test_training(input_path, output_path, file_list, train_ratio = 0.8):
    """
    """
    directories = ["train", "validation"]
    subdirectories = ["input_imges", "output_images"]

    for directory in directories: 
        os.makedirs(directory, exist_ok = True)
        for subdirectory in subdirectories: 
            os.makedirs(os.path.join(directory, subdirectory), exist_ok=True)
    
    input_images = [os.path.join(input_path, file) for file in file_list]
    output_images = [os.path.join(output_path, file) for file in file_list]

    #Make split
    input_train, input_valid, output_train, output_valid = train_test_split(input_images, output_images, train_size=train_ratio, random_state=42, shuffle=True)

    move_files(input_path, 'train/input_images', input_train)
    move_files(output_path, 'train/output_images', output_train)
    move_files(input_path, 'validation/input_images', input_valid)
    move_files(output_path, 'validation/output_images', output_valid)


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
#TODO - Look at gammas/negative vs positive
def asymmetric_loss(inputs, targets, gamma_neg=4, gamma_pos=1):
    """gamma_neg, gamma_pos: Control the focus on false negatives and false positives.
    Higher values of gamma_neg or gamma_pos will focus more on minimizing the
    corresponding type of error. For example, if false negatives are more costly than
    false positives, you might set gamma_neg higher than gamma_pos."""
    loss = -targets * torch.log(inputs + 1e-5) * torch.pow((1 - inputs), gamma_pos) - (
        1 - targets
    ) * torch.log(1 - inputs + 1e-5) * torch.pow(inputs, gamma_neg)
    return loss.mean()

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


