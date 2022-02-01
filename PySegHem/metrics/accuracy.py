import torch

def accuracy(preds,true_labels):
    preds=preds.view(-1)
    true_labels=true_labels.view(-1)
    return torch.mean(torch.sum(preds*true_labels))