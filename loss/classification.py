from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSS

@LOSS
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, logits: bool = False, reduce: bool = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(pred, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(pred, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

@LOSS
class Iterative_Focal_Loss(nn.Module):
    def __init__(self, iteration: int = 8, discount_factor: int = 0.5, alpha: float = 1.0, gamma: float = 2.0):
        super(Iterative_Focal_Loss, self).__init__()
        self.loss = FocalLoss(alpha, gamma)
        self.iteration = iteration
        self.discount_factor = discount_factor
    
    def forward(self, predictions: Dict, ground_truth: Dict) -> Dict[str, torch.Tensor]:
        assert "source_mask_pred" in predictions and "template_mask_pred" in predictions
        assert "source_mask_gt" in ground_truth and "template_mask_gt" in ground_truth
        
        source_mask_gt = ground_truth["source_mask_gt"]
        template_mask_gt = ground_truth["template_mask_gt"]
        source_mask_pred = predictions["source_mask_pred"]
        template_mask_pred = predictions["template_mask_pred"]

        losses = []
        for i in range(self.iteration):
            loss_s = self.loss(source_mask_pred[i], source_mask_gt)
            loss_t = self.loss(template_mask_pred[i], template_mask_gt)
            losses.append(self.discount_factor**(self.iteration - i) * (loss_s + loss_t))
        return {"loss": torch.sum(torch.stack(losses))}