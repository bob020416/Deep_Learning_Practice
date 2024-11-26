import torch
import torch.nn as nn

def dice_score(pred_mask, gt_mask):
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()
    intersection = torch.sum(pred_mask * gt_mask)
    union = torch.sum(pred_mask) + torch.sum(gt_mask)
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return dice.item()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_bce=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        loss_dice = self.dice_loss(inputs, targets)
        loss_bce = self.bce_loss(inputs, targets)
        return self.weight_dice * loss_dice + self.weight_bce * loss_bce
