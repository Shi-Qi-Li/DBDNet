from typing import Tuple

import numpy as np
import numpy.typing as npt

from .process import transform

def generate_overlap_mask(
        pc_1: npt.NDArray[np.float32], 
        pc_2: npt.NDArray[np.float32], 
        R: npt.NDArray[np.float32], 
        t: npt.NDArray[np.float32], 
        threshold: float = 0.1
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    r"""
    Calculate the overlap mask by using the ground truth transformation. 
    """
    
    dist = square_distance(transform(pc_1[:,:3], R, t), pc_2[:,:3])
    dist_1 = np.amin(dist, axis=1)
    dist_2 = np.amin(dist, axis=0)

    pc1_mask = (dist_1 < threshold * threshold).astype(np.float32)
    pc2_mask = (dist_2 < threshold * threshold).astype(np.float32)

    return pc1_mask, pc2_mask

def square_distance(pc1: npt.NDArray[np.float32], pc2: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    dist = -2 * np.dot(pc1, pc2.T)
    dist += np.sum(pc1 ** 2, axis=-1, keepdims=True)
    dist += np.sum(pc2 ** 2, axis=-1, keepdims=True).T

    return dist