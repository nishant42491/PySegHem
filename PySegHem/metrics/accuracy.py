import torch

def accuracy(preds,true_labels):
    """
    Computes and returns accuracy of prediction
    Args:
        preds (tensor) : predicted masks
        true_labels (tensor): target masks
    Output:
        Accuracy (tensor)
    """
    preds=preds.view(-1)
    true_labels=true_labels.view(-1)
    return torch.mean(torch.sum(preds*true_labels))