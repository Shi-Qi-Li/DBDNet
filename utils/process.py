from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import math
import copy
import open3d as o3d
import torch
import torch.nn.functional as F

from .format import covert_to_pcd

def generate_axis_rotation_matrix(theta, axis):
    if axis == 'x':
        rows = [1, 1, 2, 2]
        columns = [1, 2, 1, 2]
    elif axis == 'y':
        rows = [0, 2, 0, 2]
        columns = [0, 0, 2, 2]
    else:
        rows = [0, 0, 1, 1]
        columns = [0, 1, 0, 1]
    
    rotation_matrix = np.eye(3, dtype=np.float32)

    rotation_matrix[rows[0]][columns[0]] = math.cos(math.radians(theta))
    rotation_matrix[rows[1]][columns[1]] = -math.sin(math.radians(theta))
    rotation_matrix[rows[2]][columns[2]] = math.sin(math.radians(theta))
    rotation_matrix[rows[3]][columns[3]] = math.cos(math.radians(theta))

    return rotation_matrix

def generate_random_rotation_matrix(theta1: float = -45, theta2: float = 45) -> npt.NDArray:
    thetax, thetay, thetaz = np.random.uniform(theta1, theta2, size=(3,))
    matx = generate_axis_rotation_matrix(thetax, 'x')
    maty = generate_axis_rotation_matrix(thetay, 'y')
    matz = generate_axis_rotation_matrix(thetaz, 'z')
    return np.dot(matz, np.dot(maty, matx))

def generate_random_tranlation_vector(range1=-1, range2=1):
    tranlation_vector = np.random.uniform(range1, range2, size=(3, )).astype(np.float32)
    return tranlation_vector

def random_select_points(pc: npt.NDArray, sample_num: int, return_idx: bool = False):
    point_num = pc.shape[0]
    if point_num == sample_num or sample_num < 0:
        idx = np.arange(point_num)
        np.random.shuffle(idx)
    elif point_num > sample_num:
        idx = np.random.choice(point_num, sample_num, replace=False)
    else:
        idx = np.arange(point_num)
        np.random.shuffle(idx)
        idx_redundant = np.random.choice(point_num, sample_num - point_num, replace=True)
        idx = np.concatenate([idx, idx_redundant])

    if return_idx:
        return pc[idx, :], idx
    
    return pc[idx, :]

def transform(pc, R, t=None):
    transformed_pc = np.dot(pc, R.T)
    if t is not None:
        transformed_pc = transformed_pc + t
    return transformed_pc

def batch_transform(batch_pc, batch_R, batch_t=None):
    transformed_pc = torch.bmm(batch_pc, batch_R.permute(0, 2, 1).contiguous())
    if batch_t is not None:
        transformed_pc = transformed_pc + torch.unsqueeze(batch_t, 1)
    return transformed_pc

def batch_quat2mat(batch_quat):
    w, x, y, z = batch_quat[:, 0], batch_quat[:, 1], batch_quat[:, 2], batch_quat[:, 3]
    device = batch_quat.device
    B = batch_quat.size()[0]
    R = torch.zeros(dtype=torch.float, size=(B, 3, 3)).to(device)
    R[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    R[:, 0, 1] = 2 * x * y - 2 * z * w
    R[:, 0, 2] = 2 * x * z + 2 * y * w
    R[:, 1, 0] = 2 * x * y + 2 * z * w
    R[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    R[:, 1, 2] = 2 * y * z - 2 * x * w
    R[:, 2, 0] = 2 * x * z - 2 * y * w
    R[:, 2, 1] = 2 * y * z + 2 * x * w
    R[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    return R

def jitter(pc, sigma=0.01, clip=0.05):
    """
    Add Gaussian noise to each point.
    """
    N, C = pc.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(np.float32)
    jittered_data += pc
    return jittered_data

def uniform_jitter(pc: npt.NDArray, range: float = 0.005) -> npt.NDArray:
    """
    Add uniform noise to each point.
    """
    N, C = pc.shape
    jittered_data = range * (np.random.rand(N, C) - 0.5).astype(np.float32)
    jittered_data += pc
    return jittered_data

def random_3d_vector():
    """
    Generate a random unit 3d vector.
    """
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack((x, y, z), axis=-1)

def random_crop(pc: npt.NDArray[np.float32], keep_ratio: float = 0.7) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Random crop some points.
    """
    keep_num = round(pc.shape[0] * keep_ratio)
    dirtection = random_3d_vector()
    centered_pc = pc[:, :3] - np.mean(pc[:, :3], axis=0)
    dis = np.dot(centered_pc, dirtection)

    if keep_ratio == 0.5:
        mask = dis > 0
    else:
        mask = dis > np.percentile(dis, (1.0 - keep_ratio) * 100)

    cropped_pc = random_select_points(pc[mask, :], keep_num)

    return cropped_pc, dirtection


def batch_square_distance(batch_pc_1, batch_pc_2):
    assert batch_pc_1.shape[0] == batch_pc_2.shape[0] and batch_pc_1.shape[-1] == batch_pc_2.shape[-1]
    
    dist = -2 * torch.matmul(batch_pc_1, batch_pc_2.transpose(-1, -2))
    dist += torch.sum(batch_pc_1 ** 2, dim=-1, keepdim=True)
    dist += torch.sum(batch_pc_2 ** 2, dim=-1, keepdim=True).transpose(-1, -2)
    dist = F.relu(dist)

    return dist

def batch_euclidean_distance(batch_pc_1: torch.Tensor, batch_pc_2: torch.Tensor) -> torch.Tensor:
    square_dist = batch_square_distance(batch_pc_1, batch_pc_2)

    eps = 1e-8
    mask = (square_dist == 0.0).float()
    square_dist = square_dist + mask * eps
    dist = torch.sqrt(square_dist)

    dist = dist * (1.0 - mask)

    return dist

def farthest_point_sample(batch_pc, sample_num) -> torch.Tensor:
    device = batch_pc.device
    B, N = batch_pc.shape[0], batch_pc.shape[1]
    centroid_indices = torch.zeros(B, sample_num, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(sample_num):
        centroid_indices[:, i] = farthest
        centroid = batch_pc[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((batch_pc - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroid_indices

def ball_query(radius, sample_num, batch_pc, centroids) -> torch.Tensor:
    device = batch_pc.device
    B, N  = batch_pc.shape[0], batch_pc.shape[1]
    K = centroids.shape[1]
    
    group_indices = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, K, 1])

    square_dist = batch_square_distance(centroids, batch_pc)

    group_indices[square_dist > radius ** 2] = N
    group_indices = group_indices.sort(dim=-1)[0][:, :, :sample_num]

    group_first = group_indices[:, :, 0].view(B, K, 1).repeat([1, 1, sample_num])
    mask = group_indices == N
    group_indices[mask] = group_first[mask]
    return group_indices


def batch_get_points_by_index(batch_pc, index) -> torch.Tensor:
    device = batch_pc.device
    B = batch_pc.shape[0]
    
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1

    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    batch_new_pc = batch_pc[batch_indices, index, :]
    return batch_new_pc

def batch_get_overlap_index(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    batch_idx = torch.empty_like(mask, dtype=torch.long)

    for i in range(mask.shape[0]):
        idx = torch.nonzero(mask[i]).view(1, -1)
        count = idx.shape[-1]
        batch_idx[i][:count] = idx
        if count < batch_idx.shape[-1]:
            batch_idx[i][count:] = torch.multinomial(mask[i], batch_idx.shape[-1] - count, False)

    return batch_idx

def generate_uniform_sphere(point_num: int = 2048):
    phi = math.pi * (math.sqrt(5) - 1)
    index = torch.arange(point_num)
    y = 1 - (index / (point_num - 1)) * 2
    radius = torch.sqrt(1 - torch.square(y))

    theta = phi * index

    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius

    points = torch.cat([x.unsqueeze(dim=0), y.unsqueeze(dim=0), z.unsqueeze(dim=0)], dim=0)

    return points

def get_correspondence(
        source: Union[npt.NDArray, torch.Tensor],
        template: Union[npt.NDArray, torch.Tensor],
        R: Union[npt.NDArray, torch.Tensor],
        t: Union[npt.NDArray, torch.Tensor], 
        search_radius: float = 0.0375, 
        K: int = None
    ) -> npt.NDArray:
    if isinstance(source, torch.Tensor):
        source = source.detach().cpu().numpy()
    if isinstance(template, torch.Tensor):
        template = template.detach().cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()

    assert template.ndim == 2 and source.ndim == 2

    transformed_source = transform(copy.deepcopy(source), R, t)

    template_pcd = covert_to_pcd(template)
    pcd_tree = o3d.geometry.KDTreeFlann(template_pcd)
    
    corrs = []
    for i in range(transformed_source.shape[0]):
        point = transformed_source[i]
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, search_radius)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            corrs.append([i, j])
    return np.array(corrs)

def estimate_normals(pc: npt.NDArray) -> npt.NDArray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    return normals

def regularize_normals(pc: npt.NDArray, normals: npt.NDArray, positive: bool = True):
    """Regularize the normals towards the positive/negative direction to the origin point.

    positive: the origin point is on positive direction of the normals.
    negative: the origin point is on negative direction of the normals.
    """
    dot_products = -(pc * normals).sum(axis=1, keepdims=True)
    direction = dot_products > 0
    if positive:
        normals = normals * direction - normals * (1 - direction)
    else:
        normals = normals * (1 - direction) - normals * direction
    return normals