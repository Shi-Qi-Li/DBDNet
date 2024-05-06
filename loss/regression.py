from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .builder import LOSS
from utils import batch_inv_R_t

@LOSS
class Rotation_Regression_Loss(nn.Module):
    def __init__(self, p: int = 1, mode: str = "direct"):
        super().__init__()
        self.mode = mode
        if p == 1:
            self.loss = nn.L1Loss()
        elif p == 2:
            self.loss = nn.MSELoss()
        else:
            raise ValueError
        
    def forward(self, predictions: Dict[str, Union[torch.Tensor, Tuple]], ground_truth: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "R_pred" in predictions and "R_gt" in ground_truth
        
        R_pred = predictions["R_pred"]
        R_gt = ground_truth["R_gt"]

        if self.mode == "direct":    
            if isinstance(R_pred, torch.Tensor):
                assert R_pred.shape == R_gt.shape
                loss = self.loss(R_pred, R_gt)
            elif isinstance(R_pred, Tuple):
                loss = 0.0
                for i in range(len(R_pred)):
                    assert R_pred[i].shape == R_gt.shape
                    loss = loss + self.loss(R_pred[i], R_gt)
                loss = loss / len(R_pred)
        elif self.mode == "cycle":
            inv_R_gt, _ = batch_inv_R_t(ground_truth["R_gt"])
            if isinstance(R_pred, torch.Tensor):
                
                assert R_pred.shape == inv_R_gt.shape
                
                B = R_pred.shape[0]
                eye = torch.eye(3, device=R_pred.device).unsqueeze_(dim=0).repeat(B, 1, 1)

                loss = self.loss(torch.bmm(inv_R_gt, R_pred), eye)
            elif isinstance(R_pred, Tuple):
                loss = 0.0
                B = R_pred[0].shape[0]
                eye = torch.eye(3, device=R_pred[0].device).unsqueeze_(dim=0).repeat(B, 1, 1)
                for i in range(len(R_pred)):
                    assert R_pred[i].shape == inv_R_gt.shape

                    loss = loss + self.loss(torch.bmm(inv_R_gt, R_pred[i]), eye)
                loss = loss / len(R_pred)
        else:
            raise NotImplementedError
        
        return {"loss": loss}

@LOSS
class Translation_Regression_Loss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        if p == 1:
            self.loss = nn.SmoothL1Loss()
        elif p == 2:
            self.loss = nn.MSELoss()
        else:
            raise ValueError
        
    def forward(self, predictions: Dict, ground_truth: Dict) -> Dict[str, torch.Tensor]:
        assert "t_pred" in predictions and "t_gt" in ground_truth
        
        t_pred = predictions["t_pred"]
        t_gt = ground_truth["t_gt"]
        
        if isinstance(t_pred, torch.Tensor):
            assert t_pred.shape == t_gt.shape
            loss = self.loss(t_pred, t_gt)
        elif isinstance(t_pred, Tuple):
            loss = 0.0
            for i in range(len(t_pred)):
                assert t_pred[i].shape == t_gt.shape    
                loss = loss + self.loss(t_pred[i], t_gt)
            loss = loss / len(t_pred)

        return {"loss": loss}