from torchvision.ops import sigmoid_focal_loss
import torch.nn as nn
from typing import Optional


class DiceLoss(nn.Module):
    def forward(self, input, target):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        return sigmoid_focal_loss(input, target, self.alpha, self.gamma)


class CombinedLoss(nn.Module):
    def __init__(self, cross: int = 0.34, dice: Optional[int] = 0.33, focal: Optional[int] = 0.33):
        super().__init__()
        self.alpha = dice
        self.beta = focal
        self.gamma = cross
        assert (self.beta + self.alpha + self.gamma == 1), "The sum of the weights should be equal to 1"
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss1 = self.dice_loss(input, target) if self.alpha else 0
        loss2 = self.focal_loss(input, target) if self.beta else 0
        loss3 = self.cross_entropy(input, target)
        return self.alpha * loss1 + self.beta * loss2 + self.gamma * loss3
