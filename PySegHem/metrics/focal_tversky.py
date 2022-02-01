import torch


def tversky_index(yhat, ytrue, alpha=0.3, beta=0.7, epsilon=1e-6):
    """
    Computes Tversky index
    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight for False positive
        beta (Float): weight for False negative
                    `` alpha and beta control the magnitude of penalties and should sum to 1``
        epsilon (Float): smoothing value to avoid division by 0
    output:
        tversky index value
    """
    TP = torch.sum(yhat * ytrue, (1, 2, 3))
    FP = torch.sum((1. - ytrue) * yhat, (1, 2, 3))
    FN = torch.sum((1. - yhat) * ytrue, (1, 2, 3))

    return TP / (TP + alpha * FP + beta * FN + epsilon)

def Tversky_Focal_Loss(yhat, ytrue, alpha=0.7, beta=0.3, gamma=0.75):
    """
    Computes tversky focal loss for highly umbalanced data
    https://arxiv.org/pdf/1810.07842.pdf
    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight for False positive
        beta (Float): weight for False negative
                    `` alpha and beta control the magnitude of penalties and should sum to 1``
        gamma (Float): focal parameter
                    ``control the balance between easy background and hard ROI training examples``
    output:
        tversky focal loss value with `mean` reduction
    """

    return torch.mean(torch.pow(1 - tversky_index(yhat, ytrue, alpha, beta), gamma))