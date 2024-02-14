def evaluate(y_pred, y_true, epsilon: float=1e-10): 
    """
    epsilon is a small number that is added to avoid potential division by 0 
    """
    TP = 0 #True positive
    FP = 0 #False positive
    FN = 0 #False negative

    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)

    F1 = 2 * (precision*recall)/(precision+recall+epsilon)
    
    
    return precision, recall, F1

def violin_plot(): 
    return

