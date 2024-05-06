from typing import Union

import random
import torch
import numpy as np
import open3d as o3d

def covert_to_pcd(origin: Union[torch.Tensor, np.ndarray], ind: int = -1):
    """
    Covert numpy points to Open3D point format.
    """
    colors = [[1.0, 0, 0], # red
              [0, 1.0, 0], # green
              [0, 0, 1.0], # blue
              [0, 0, 0], # black
              [1, 0.706, 0], # source
              [0, 0.651, 0.929]] # template 
    color = colors[ind] if 0 < ind and ind < 6 else [random.random() for _ in range(3)]
    pcd = o3d.geometry.PointCloud()
    
    if isinstance(origin, np.ndarray):
        pcd.points = o3d.utility.Vector3dVector(origin)
    elif isinstance(origin, torch.Tensor):
        pcd.points = o3d.utility.Vector3dVector(origin.cpu().detach().numpy())
    else:
        raise NotImplementedError

    if ind >= 0:
        pcd.paint_uniform_color(color)
    return pcd

def covert_to_tpcd(origin: Union[torch.Tensor, np.ndarray], ind: int = -1):
    """
    Covert numpy points to Open3D point format.
    """
    colors = [[1.0, 0, 0], # red
              [0, 1.0, 0], # green
              [0, 0, 1.0], # blue
              [0, 0, 0]] # black
    color = colors[ind] if ind < 4 else [random.random() for _ in range(4)]
      
    if isinstance(origin, np.ndarray):
        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(origin))
    elif isinstance(origin, torch.Tensor):
        pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(origin.cpu().detach().numpy()))
    else:
        raise NotImplementedError

    if ind >= 0:
        pcd.paint_uniform_color(color)
    return pcd