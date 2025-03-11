import torch
import torch.nn as nn


class PCSRLoss(nn.Module):
    def __init__(self):
        super(PCSRLoss, self).__init__()
        
    def forward(self, y_pred, y, mask):
        # loss = 1 - ((2 * y_pred - 1) * (2 * y - 1)).mean()
        # loss = nn.functional.binary_cross_entropy(y_pred, y)

            # 计算每个位置的 BCE 损失（未聚合）
        bce = nn.functional.binary_cross_entropy(y_pred, y, reduction='none')  # 保持逐元素损失
        loss = bce[mask.type(torch.bool)].mean()
        
        return loss
