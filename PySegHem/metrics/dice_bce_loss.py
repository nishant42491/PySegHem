import torch
import torch.nn.functional as F


def Dice_BCE_Loss(inputs, targets, smooth=1):
    """
    Combines Binary Cross Entropy with Dice Loss Function
    Args:
        inputs (tensor): predicted masks
        targets (tensor): targets masks
        smooth (int): (default=1)
    Output:
        dice bce coefficient (tensor)
    """
    # comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    Dice_BCE = BCE + dice_loss

    return Dice_BCE