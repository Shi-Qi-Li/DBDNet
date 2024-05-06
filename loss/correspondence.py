from typing import Dict, Tuple

import torch
import torch.nn as nn

from .builder import LOSS
from utils import batch_transform, batch_get_points_by_index

@LOSS
class Iterative_Distance_Loss(nn.Module):
    def __init__(self, iteration: int = 8, discount_factor: int = 0.5):
        super(Iterative_Distance_Loss, self).__init__()
        self.iteration = iteration
        self.discount_factor = discount_factor
        self.loss = nn.L1Loss()
    
    def forward(self, predictions: Dict, ground_truth: Dict) -> Dict[str, torch.Tensor]:
        assert "correspondences" in predictions
        assert all(key in ground_truth for key in ("R_gt", "t_gt", "source_pc", "template_pc"))

        pred_correspondences = predictions["correspondences"]

        template_pc, source_pc = ground_truth["template_pc"], ground_truth["source_pc"]

        if "source_idx_pred" in predictions and "template_idx_pred" in predictions:
            source_pc = batch_get_points_by_index(source_pc, predictions["source_idx_pred"])
            template_pc = batch_get_points_by_index(template_pc, predictions["template_idx_pred"])

        R_gt, t_gt = ground_truth["R_gt"], ground_truth["t_gt"]
        
        transformed_source = batch_transform(source_pc[...,:3], R_gt, t_gt)

        losses = []
        discount_factor = 0.5
        for i in range(self.iteration):
            if isinstance(pred_correspondences[i], torch.Tensor):
                permutated_template = torch.bmm(pred_correspondences[i], template_pc)
                loss = self.loss(permutated_template, transformed_source)
            elif isinstance(pred_correspondences[i], Tuple):
                loss = 0.0
                for pred_correspondence in pred_correspondences[i]:
                    permutated_template = torch.bmm(pred_correspondence, template_pc)
                    loss = loss + self.loss(permutated_template, transformed_source)
                loss = loss / len(pred_correspondences[i])
                
            losses.append(discount_factor**(self.iteration - i)*loss)
        return {"loss": torch.sum(torch.stack(losses))}