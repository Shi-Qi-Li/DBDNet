from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .builder import LOSS
from utils import batch_transform


@LOSS
class Regression_Distance_Loss(nn.Module):
    def __init__(self):
        super(Regression_Distance_Loss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, predictions: Dict[str, Union[torch.Tensor, Tuple[torch.Tensor]]], ground_truth: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "R_pred" in predictions and "t_pred" in predictions
        assert all(key in ground_truth for key in ("R_gt", "t_gt", "source_pc"))

        source_pc = ground_truth["source_pc"]

        R_gt, t_gt = ground_truth["R_gt"], ground_truth["t_gt"]
        R_pred, t_pred = predictions["R_pred"], predictions["t_pred"]
        
        transformed_source_gt = batch_transform(source_pc, R_gt, t_gt)
        
        if isinstance(R_pred, torch.Tensor):
            transformed_source_pred = batch_transform(source_pc, R_pred, t_pred)
            loss = torch.mean(self.loss(transformed_source_pred, transformed_source_gt))
        elif isinstance(R_pred, Tuple):
            assert len(R_pred) == len(t_pred), "R and t prediction number must be equal"
            loss = 0.0
            for i in range(len(R_pred)):
                transformed_source_pred = batch_transform(source_pc, R_pred[i], t_pred[i])
                loss = loss + torch.mean(self.loss(transformed_source_pred, transformed_source_gt))
            loss = loss / len(R_pred)

        return {"loss": loss}