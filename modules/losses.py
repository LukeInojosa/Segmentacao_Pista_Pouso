import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()
    def forward(self,y_hat,y):

        y = y.flatten(1)

        y_hat = torch.sigmoid(y_hat).flatten(1)


        intersect = (y * y_hat).sum(dim=1)
        union = y.sum(dim=1) + y_hat.sum(dim=1)

        dice = (2 * intersect)/(union+ 1e-8)

        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss,self).__init__()
    def forward(self,y_hat,y):
        return sigmoid_focal_loss(y_hat,y, reduction='mean', alpha=0.75)