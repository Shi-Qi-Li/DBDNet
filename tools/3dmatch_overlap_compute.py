from typing import Tuple

import os
import pickle
import argparse
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import open3d as o3d
import torch
import sys
sys.path.append(os.getcwd())

from utils import batch_square_distance, batch_transform, transform, covert_to_pcd


def config_params():
    parser = argparse.ArgumentParser(description='Pre compute Parameters')
    parser.add_argument('--data_path', default="data/3dmatch/indoor", help='the data path')
    
    args = parser.parse_args()
    return args

def generate_overlap_mask_cuda(
        pc_1: npt.NDArray[np.float32], 
        pc_2: npt.NDArray[np.float32], 
        R: npt.NDArray[np.float32], 
        t: npt.NDArray[np.float32], 
        threshold: float = 0.1
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    r"""
    Calculate the overlap mask by using the ground truth transformation using torch. 
    """
    
    pc_1 = torch.tensor(pc_1, device="cpu").unsqueeze_(dim=0)
    pc_2 = torch.tensor(pc_2, device="cpu").unsqueeze_(dim=0)
    R = torch.tensor(R, device="cpu").unsqueeze_(dim=0)
    t = torch.tensor(t, device="cpu").unsqueeze_(dim=0)
    
    dist = batch_square_distance(batch_transform(pc_1, R, t), pc_2)

    dist_1 = torch.min(dist[0], dim=1)[0]
    dist_2 = torch.min(dist[0], dim=0)[0]

    pc1_mask = (dist_1 < threshold * threshold).cpu().numpy().astype(np.float32)
    pc2_mask = (dist_2 < threshold * threshold).cpu().numpy().astype(np.float32)

    return pc1_mask, pc2_mask

def generate_correspondence(
        pc_1: npt.NDArray[np.float32], 
        pc_2: npt.NDArray[np.float32], 
        R: npt.NDArray[np.float32], 
        t: npt.NDArray[np.float32],
        threshold: float = 0.0375
    ):

    pc_1 = transform(pc_1, R, t)

    pcd_1 = covert_to_pcd(pc_1)
    pcd_2 = covert_to_pcd(pc_2)

    # Check which points in tgt has a correspondence (i.e. point nearby) in the src,
    # and then in the other direction. As long there's a point nearby, it's
    # considered to be in the overlap region. For correspondences, we require a stronger
    # condition of being mutual matches
    tgt_corr = np.full(pc_2.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_1)
    for i, t in enumerate(pc_2):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(t, threshold)
        if num_knn > 0:
            tgt_corr[i] = knn_indices[0]
    src_corr = np.full(pc_1.shape[0], -1)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_2)
    for i, s in enumerate(pc_1):
        num_knn, knn_indices, knn_dist = pcd_tree.search_radius_vector_3d(s, threshold)
        if num_knn > 0:
            src_corr[i] = knn_indices[0]

    # Compute mutual correspondences
    src_corr_is_mutual = np.logical_and(tgt_corr[src_corr] == np.arange(len(src_corr)),
                                        src_corr > 0)
    src_tgt_corr = np.stack([np.nonzero(src_corr_is_mutual)[0],
                             src_corr[src_corr_is_mutual]])

    has_corr_src = src_corr >= 0
    has_corr_tgt = tgt_corr >= 0

    return has_corr_src, has_corr_tgt, src_tgt_corr

def pre_compute_overlap(data_path: str, split: str):
    overlap_radius = 0.0375
    info_file = os.path.join(data_path, '{}.pkl'.format(split))
    
    with open(info_file, "rb") as f:
        info_data = pickle.load(f)

        R = info_data["rot"]
        t = info_data["trans"]
        source_path = info_data["src"]
        template_path = info_data["tgt"]

    data_loop = tqdm(range(len(source_path)))
    data_loop.set_description("Pre compute {} overlap".format(split))
    
    source_masks = []
    template_masks = []
    correspondences = []
    
    for idx in data_loop:
        source_pc = torch.load(f = os.path.join(data_path, source_path[idx]))
        template_pc = torch.load(f = os.path.join(data_path, template_path[idx]))
        
        source_mask, template_mask = generate_overlap_mask_cuda(source_pc, template_pc, R[idx], t[idx].reshape(3,), overlap_radius)
        
        source_masks.append(source_mask)
        template_masks.append(template_mask)
        
    mask_file = os.path.join(data_path, '{}_overlap_info.pkl'.format(split))
    
    data = {
        "source_mask": source_masks,
        "template_mask": template_masks,
        "correspondence": correspondences
    }
    
    with open(mask_file, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    args = config_params()
    
    splits = ["valid", "train", "3DMatch", "3DLoMatch"]

    for split in splits:
        pre_compute_overlap(args.data_path, split)