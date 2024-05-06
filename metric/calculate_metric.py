from typing import Dict

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from utils import batch_inv_R_t


def anisotropic_R_error(R_pred, R_gt):
    assert R_pred.shape == R_gt.shape
    if isinstance(R_pred, torch.Tensor):
        R_pred = R_pred.cpu().detach().numpy()
    if isinstance(R_gt, torch.Tensor):
        R_gt = R_gt.cpu().detach().numpy()
    eulers_pred, eulers_gt = [], []
    for i in range(R_pred.shape[0]):
        eulers_pred.append(Rotation.from_matrix(R_pred[i]).as_euler(seq="xyz", degrees=True))
        eulers_gt.append(Rotation.from_matrix(R_gt[i]).as_euler(seq="xyz", degrees=True))
    eulers_pred = np.stack(eulers_pred, axis=0)
    eulers_gt = np.stack(eulers_gt, axis=0)
    R_mse = np.mean((eulers_pred - eulers_gt) ** 2, axis=-1)
    R_mae = np.mean(np.abs(eulers_pred - eulers_gt), axis=-1)
    return R_mse, R_mae

def anisotropic_t_error(t_pred, t_gt):
    assert t_pred.shape == t_gt.shape
    if isinstance(t_pred, torch.Tensor):
        t_pred = t_pred.cpu().detach().numpy()
    if isinstance(t_gt, torch.Tensor):
        t_gt = t_gt.cpu().detach().numpy()
    t_mse = np.mean((t_pred - t_gt) ** 2, axis=-1)
    t_mae = np.mean(np.abs(t_pred - t_gt), axis=-1)
    return t_mse, t_mae

def isotropic_R_error(R_pred, R_gt):
    R_gt_inv = R_gt.permute(0, 2, 1).contiguous()
    I = torch.bmm(R_gt_inv, R_pred)
    tr = I[:, 0, 0] + I[:, 1, 1] + I[:, 2, 2]
    radians = torch.acos(torch.clamp((tr - 1) / 2, min=-1, max=1))
    degrees = radians.rad2deg()
    return degrees.cpu().detach().numpy()

def isotropic_t_error(t_pred, R_gt, t_gt):
    R_gt, t_gt = batch_inv_R_t(R_gt, t_gt)
    inv_t_pred = torch.bmm(R_gt, t_pred.unsqueeze(-1)).squeeze(-1)
    return torch.norm(inv_t_pred + t_gt, p=2, dim=-1).cpu().detach().numpy()

def classification_error(mask_pred, mask_gt):
    tp = ((mask_pred >= 0.5) & (mask_gt == 1)).sum(dim=-1).cpu().detach().numpy()
    fp = ((mask_pred >= 0.5) & (mask_gt == 0)).sum(dim=-1).cpu().detach().numpy()
    fn = ((mask_pred < 0.5) & (mask_gt == 1)).sum(dim=-1).cpu().detach().numpy()
    tn = ((mask_pred < 0.5) & (mask_gt == 0)).sum(dim=-1).cpu().detach().numpy()

    return tp, fp, fn, tn

def compute_metrics(predictions: Dict, ground_truth: Dict) -> Dict:
    metrics = dict()

    if all(key in predictions for key in ('R_pred', 't_pred')) and all(key in ground_truth for key in ('R_gt', 't_gt')):
        R_pred, t_pred = predictions["R_pred"], predictions["t_pred"]
        R_gt, t_gt = ground_truth["R_gt"], ground_truth["t_gt"]
        
        metrics["R_mse"], metrics["R_mae"] = anisotropic_R_error(R_pred, R_gt)
        metrics["t_mse"], metrics["t_mae"] = anisotropic_t_error(t_pred, t_gt)
        metrics["R_isotropic"] = isotropic_R_error(R_pred, R_gt)
        metrics["t_isotropic"] = isotropic_t_error(t_pred, R_gt ,t_gt)

    if all(key in predictions for key in ('template_mask_pred', 'source_mask_pred')) and all(key in ground_truth for key in ('template_mask_gt', 'source_mask_gt')):
        template_mask_preds = predictions["template_mask_pred"]
        source_mask_preds = predictions["source_mask_pred"]
        template_mask_gt = ground_truth["template_mask_gt"]
        source_mask_gt = ground_truth["source_mask_gt"]

        confusion_matrix = (0, 0, 0, 0)
        for template_mask_pred, source_mask_pred in zip(template_mask_preds, source_mask_preds):
            source_error = classification_error(source_mask_pred, source_mask_gt)
            confusion_matrix = tuple(map(sum, zip(confusion_matrix, source_error)))
            template_error = classification_error(template_mask_pred, template_mask_gt)
            confusion_matrix = tuple(map(sum, zip(confusion_matrix, template_error)))

        metrics["tp"] = confusion_matrix[0]
        metrics["fp"] = confusion_matrix[1]
        metrics["fn"] = confusion_matrix[2]
        metrics["tn"] = confusion_matrix[3]

    if "logits" in predictions and "label" in ground_truth:
        metrics["pred"] = predictions["logits"].max(dim=1)[1].squeeze_().detach().cpu().numpy()
        metrics["label"] = ground_truth["label"].squeeze_().cpu().numpy()
    
    return metrics