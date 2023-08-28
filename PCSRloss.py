import torch.nn as nn


class PCSRLoss(nn.Module):
    def __init__(self):
        super(PCSRLoss, self).__init__()
        
    def forward(self, y_pred, y):
        # loss = 1 - ((2 * y_pred - 1) * (2 * y - 1)).mean()
        loss = nn.functional.binary_cross_entropy(y_pred, y)
        
        return loss
