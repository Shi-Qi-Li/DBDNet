from typing import List, Dict, Union

import torch
import numpy as np
import open3d as o3d
from utils import covert_to_pcd, batch_transform


def visualization(pcs: List[Union[torch.Tensor, np.ndarray]]):
    pcds = []
    
    for i, pc in enumerate(pcs):
        if isinstance(pc, torch.Tensor):
            pc = torch.squeeze(pc).cpu().detach().numpy()
        
        if pc.ndim > 2:
            pc = pc[0]

        pcds.append(covert_to_pcd(pc, i))
    
    o3d.visualization.draw_geometries(pcds)

def registration_visualization(
        batch_data: Dict,
        predictions: Dict, 
        ground_truth: Dict,
        index: int = 0
    ):

    R, t = ground_truth["R_gt"], ground_truth["t_gt"]

    if "source_raw" in batch_data and "template_raw" in batch_data:
        source_pc = batch_data["source_raw"]
        template_pc = batch_data["template_raw"]
    else:    
        source_pc = batch_data["source_pc"]
        template_pc = batch_data["template_pc"]
    
    transformed_source_pc = batch_transform(source_pc.clone(), R, t)[index]
    template_pc = template_pc[index]
    source_pc = source_pc[index]

    pcds = [
        covert_to_pcd(template_pc.cpu().detach().numpy(), 5),
        covert_to_pcd(transformed_source_pc.cpu().detach().numpy(), 4)
    ]
    o3d.visualization.draw_geometries(pcds)
    
    pcds = [
        covert_to_pcd(template_pc.cpu().detach().numpy(), 5),
        covert_to_pcd(source_pc.cpu().detach().numpy(), 4)
    ]
    o3d.visualization.draw_geometries(pcds)


def overlap_visualization(
        batch_data: Dict,
        predictions: Dict, 
        ground_truth: Dict, 
        index: int = 0
    ):
    R, t = ground_truth["R_gt"], ground_truth["t_gt"]

    source_pc = batch_data["source_pc"]
    template_pc = batch_data["template_pc"]
    
    source_pc = batch_transform(source_pc, R, t)
    template_pc = template_pc[index]
    source_pc = source_pc[index]
    
    
    template_mask_preds = predictions["template_mask_pred"][index]
    source_mask_preds = predictions["source_mask_pred"][index]
    template_mask_gt = ground_truth["template_mask_gt"][index]
    source_mask_gt = ground_truth["source_mask_gt"][index]

    pcds = []
    
    template_tp_indices = ((template_mask_preds >= 0.5) & (template_mask_gt == 1)).cpu().detach().numpy()
    template_tp = template_pc[template_tp_indices].cpu().detach().numpy()
    print(type(template_tp), template_tp.shape)
    pcds.append(covert_to_pcd(template_tp, 1)) # green

    template_fp_indices = ((template_mask_preds >= 0.5) & (template_mask_gt == 0)).cpu().detach().numpy()
    template_fp = template_pc[template_fp_indices].cpu().detach().numpy()
    pcds.append(covert_to_pcd(template_fp, 0)) # red
    
    template_fn_indices = ((template_mask_preds < 0.5) & (template_mask_gt == 1)).cpu().detach().numpy()
    template_fn = template_pc[template_fn_indices].cpu().detach().numpy()
    pcds.append(covert_to_pcd(template_fn, 2)) # blue
    
    template_tn_indices = ((template_mask_preds < 0.5) & (template_mask_gt == 0)).cpu().detach().numpy()
    template_tn = template_pc[template_tn_indices].cpu().detach().numpy()
    pcds.append(covert_to_pcd(template_tn, 3)) # black

    source_tp_indices = ((source_mask_preds >= 0.5) & (source_mask_gt == 1)).cpu().detach().numpy()
    source_tp = source_pc[source_tp_indices].cpu().detach().numpy()
    pcds.append(covert_to_pcd(source_tp, 1))

    source_fp_indices = ((source_mask_preds >= 0.5) & (source_mask_gt == 0)).cpu().detach().numpy()
    source_fp = source_pc[source_fp_indices].cpu().detach().numpy()
    pcds.append(covert_to_pcd(source_fp, 0))

    source_fn_indices = ((source_mask_preds < 0.5) & (source_mask_gt == 1)).cpu().detach().numpy()
    source_fn = source_pc[source_fn_indices].cpu().detach().numpy()
    pcds.append(covert_to_pcd(source_fn, 2))

    source_tn_indices = ((source_mask_preds < 0.5) & (source_mask_gt == 0)).cpu().detach().numpy()
    source_tn = source_pc[source_tn_indices].cpu().detach().numpy()
    pcds.append(covert_to_pcd(source_tn, 3))

    o3d.visualization.draw_geometries(pcds)