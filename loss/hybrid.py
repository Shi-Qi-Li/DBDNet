from typing import Dict

import torch
import torch.nn as nn

from .builder import LOSS, build_loss


@LOSS
class HybridLoss(nn.Module):
    def __init__(self, loss_args):
        super(HybridLoss, self).__init__()

        self.losses = []
        self.weights = []
        for loss_cfg in loss_args.values():
            weight = loss_cfg.pop("weight")
            if weight > 0:
                self.weights.append(weight)
                self.losses.append(build_loss(loss_cfg))

    def forward(self, predictions: Dict, ground_truth: Dict) -> Dict[str, torch.Tensor]:
        loss_dict = dict()
        loss_dict["loss"] = 0.0

        for weight, loss_func in zip(self.weights, self.losses):
            loss = loss_func(predictions, ground_truth)["loss"] * weight
            loss_dict["loss"] += loss
            loss_dict[loss_func.__class__.__name__] = loss

        return loss_dict