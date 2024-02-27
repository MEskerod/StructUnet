import torch



def evaluate(y_pred, y_true, epsilon: float=1e-10): 
    """
    epsilon is a small number that is added to avoid potential division by 0 
    """
    
    TP = torch.sum(y_true * y_pred) #True positive
    FP = torch.sum((1 - y_true) * y_pred) #False postive
    FN = torch.sum(y_true * (1-y_pred)) #False negative

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)

    F1 = 2 * (precision*recall)/(precision+recall+epsilon)
    
    
    return precision.item(), recall.item(), F1.item()   

def violin_plot(): 
    return

