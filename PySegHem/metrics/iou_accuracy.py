import torch

def iou_accuracy(yhat, ytrue, threshold=0.5, epsilon=1e-6):
    """
    Computes Intersection over Union metric
    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        threshold (Float): threshold for pixel classification
        epsilon (Float): smoothing parameter for numerical stability
    output:
        iou value with `mean` reduction
    """
    intersection = ((yhat>threshold).long() & ytrue.long()).float().sum((1,2,3))
    union = ((yhat>threshold).long() | ytrue.long()).float().sum((1,2,3))

    return torch.mean(intersection/(union + epsilon)).item()