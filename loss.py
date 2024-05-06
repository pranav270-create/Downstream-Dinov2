from torchvision.ops import sigmoid_focal_loss
import torch.nn as nn
import torch
from typing import Optional


def diceloss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss = torch.jit.script(
    diceloss
)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        return sigmoid_focal_loss(input, target, self.alpha, self.gamma)


class CombinedLoss(nn.Module):
    def __init__(self, cross: int = 0.34, dice: Optional[int] = 0.33, focal: Optional[int] = 0.33, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = dice
        self.beta = focal
        self.gamma = cross
        assert (self.beta + self.alpha + self.gamma == 1), "The sum of the weights should be equal to 1"
        self.dice_loss = batch_dice_loss
        self.focal_loss = FocalLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weights) if weights else nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss1 = self.dice_loss(input, target) if self.alpha else 0
        loss2 = self.focal_loss(input, target) if self.beta else 0
        loss3 = self.cross_entropy(input, target)
        return self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3
