from typing import Union, Tuple, List

import torch
import torch.nn as nn

from pointnet_ops import pointnet_utils
from pointnet2_ops import pointnet2_utils
from utils import batch_get_points_by_index

def get_activation(**act_cfg: dict) -> Union[nn.Module, None]:
    if act_cfg["type"] is None or act_cfg["type"].lower() == "none":
        return None
    
    if act_cfg["type"].lower() == 'gelu':
        del act_cfg["type"]
        return nn.GELU(**act_cfg)
    elif act_cfg["type"].lower() == 'rrelu':
        del act_cfg["type"]
        return nn.RReLU(**act_cfg)
    elif act_cfg["type"].lower() == 'selu':
        del act_cfg["type"]
        return nn.SELU(**act_cfg)
    elif act_cfg["type"].lower() == 'silu':
        del act_cfg["type"]
        return nn.SiLU(**act_cfg)
    elif act_cfg["type"].lower() == 'hardswish':
        del act_cfg["type"]
        return nn.Hardswish(**act_cfg)
    elif act_cfg["type"].lower() == 'leakyrelu':
        del act_cfg["type"]
        return nn.LeakyReLU(**act_cfg)
    elif act_cfg["type"].lower() == 'tanh':
        del act_cfg["type"]
        return nn.Tanh(**act_cfg)
    else:
        del act_cfg["type"]
        return nn.ReLU(**act_cfg)

def get_nomalization(**norm_cfg: dict) -> Union[nn.Module, None]:
    if norm_cfg["type"] is None or norm_cfg["type"].lower() == "none":
        return None
    
    if norm_cfg["type"].lower() == "bn1d":
        del norm_cfg["type"]
        return nn.BatchNorm1d(**norm_cfg)
    elif norm_cfg["type"].lower() == "bn2d":
        del norm_cfg["type"]
        return nn.BatchNorm2d(**norm_cfg)
    elif norm_cfg["type"].lower() == "ln":
        del norm_cfg["type"]
        return nn.LayerNorm(**norm_cfg)
    elif norm_cfg["type"].lower() == "gn":
        del norm_cfg["type"]
        return nn.GroupNorm(**norm_cfg)
    else:
        raise NotImplementedError


class Conv1dModule(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, norm_cfg: dict = dict(type = "None"), act_cfg: dict = dict(type = "relu", inplace = True)):
        super(Conv1dModule, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.norm = get_nomalization(**norm_cfg)
        self.act = get_activation(**act_cfg)

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x    

class Conv2dModule(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, norm_cfg: dict = dict(type = "None"), act_cfg: dict = dict(type = "relu", inplace = True)):
        super(Conv2dModule, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.norm = get_nomalization(**norm_cfg)
        self.act = get_activation(**act_cfg)

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x 

class TranslationDown(nn.Module):
    def __init__(self, k: int, dense_channe: int, sparse_channel: int):
        super(TranslationDown, self).__init__()
        self.k = k
        self.conv = Conv2dModule(dense_channe, sparse_channel)
    
    def forward(self, pc: torch.Tensor, feature: torch.Tensor):
        B, C, N = feature.shape
        sample_num = N // 2
        pc = pc.permute(0, 2, 1)
        offset = (torch.arange(B, device=pc.device, dtype=torch.int) + 1) * N

        fps_indices = pointnet2_utils.furthest_point_sample(pc.contiguous(), sample_num).to(torch.long)
        fps_pc = batch_get_points_by_index(pc, fps_indices)
        
        
        fps_offset, count = [offset[0].item() // 2], offset[0].item() // 2
        for i in range(1, offset.shape[0]):
            count += (offset[i].item() - offset[i-1].item()) // 2
            fps_offset.append(count)
        fps_offset = torch.tensor(fps_offset, device=offset.device, dtype=torch.int)

        feature = feature.permute(0, 2, 1).reshape(B * N, C)
        pc = pc.reshape(B * N, -1)
        fps_pc = fps_pc.reshape(B * sample_num, -1)
        fps_feature, _ = pointnet_utils.knn_query_and_group(feature.contiguous(), pc.contiguous(), offset, fps_pc.contiguous(), fps_offset, nsample=self.k, with_xyz=False)
        fps_feature = fps_feature.reshape(B, sample_num, self.k, -1)
        fps_feature = fps_feature.permute(0, 3, 1, 2)

        fps_feature = self.conv(fps_feature)
        fps_feature = fps_feature.max(dim=-1, keepdim=False)[0]

        fps_pc = fps_pc.reshape(B, sample_num, -1).permute(0, 2, 1)
        
        return fps_pc, fps_feature, fps_indices

class TranslationUp(nn.Module):
    def __init__(self, sparse_channel: int, dense_channe: int):
        super(TranslationUp, self).__init__()
        self.conv = Conv1dModule(sparse_channel, dense_channe)

    def forward(self, sparse_pc: torch.Tensor, dense_pc: torch.Tensor, sparse_feature: torch.Tensor) -> torch.Tensor:
        dist, idx = pointnet2_utils.three_nn(dense_pc.permute(0, 2, 1).contiguous(), sparse_pc.permute(0, 2, 1).contiguous())
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_feature = pointnet2_utils.three_interpolate(sparse_feature, idx, weight)

        interpolated_feature = self.conv(interpolated_feature)

        return interpolated_feature

class QueryDown(nn.Module):
    def __init__(self, dense_channe: int, sparse_channel: int):
        super(QueryDown, self).__init__()
        self.conv = Conv1dModule(dense_channe, sparse_channel)
    
    def forward(self, feature: torch.Tensor, indices: Union[torch.Tensor, List[torch.Tensor]]):
        feature = feature.permute(0, 2, 1)
        if isinstance(indices, torch.Tensor):
            fps_feature = batch_get_points_by_index(feature, indices)
        else:
            for index in indices:
                fps_feature = batch_get_points_by_index(feature, index)
        
        fps_feature = self.conv(fps_feature.permute(0, 2, 1))

        return fps_feature
    
class Self_Cross_Attention(nn.Module):
    def __init__(self, embedding_dim: int = 256) -> None:
        super(Self_Cross_Attention, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.self_layer_norm = nn.LayerNorm(embedding_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.cross_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, features: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        source_feature, template_feature = features[0], features[1]

        source_embedding, _ = self.self_attention(source_feature, source_feature, source_feature)
        template_embedding, _ = self.self_attention(template_feature, template_feature, template_feature)

        source_feature = self.self_layer_norm(source_feature + source_embedding)
        template_feature = self.self_layer_norm(template_feature + template_embedding)

        source_embedding, _ = self.cross_attention(source_feature, template_feature, template_feature)
        template_embedding, _ = self.cross_attention(template_feature, source_feature, source_feature)

        source_feature = self.cross_layer_norm(source_feature + source_embedding)
        template_feature = self.cross_layer_norm(template_feature + template_embedding)

        return source_feature, template_feature