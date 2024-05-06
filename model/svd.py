from typing import Tuple, Optional

import random
import torch
import torch.nn as nn


class Drop(nn.Module):
    def __init__(self, keep_ratio: float = 0.5) -> None:
        super(Drop, self).__init__()
        self.keep_ratio = keep_ratio

    def forward(self, source: torch.Tensor, template: torch.Tensor, weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = source.shape[1]
        index = torch.tensor(random.sample(range(N), int(N * self.keep_ratio)), dtype=torch.long, device=source.device)
        source = source.index_select(dim=1, index=index)
        template = template.index_select(dim=1, index=index)
        if weights != None:
            weights = weights.index_select(dim=1, index=index)

        return source, template, weights

class SVDHead(nn.Module):
    def __init__(self, eps: float = 0.0, calculate_R: bool = True, calculate_t: bool = True, keep_ratio: float = 1.0):
        super(SVDHead, self).__init__()
        self.calculate_R = calculate_R
        self.calculate_t = calculate_t
        self.svd_eps = nn.Parameter(torch.eye(3), requires_grad=False) * eps if eps > 0 else None
        self.keep_ratio = keep_ratio
        self.drop = Drop(keep_ratio) if keep_ratio < 1.0 else None

    def forward(self, source: torch.Tensor, template: torch.Tensor, weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert source.shape == template.shape, "source shape {source_shape} can't match template shape {template_shape}".format(source_shape=source.shape, template_shape=template.shape)

        if self.training and self.keep_ratio < 1.0:
            source, template, weights = self.drop(source, template, weights)
        
        batch = source.shape[0]
        if weights == None:
            weights = torch.ones((batch, source.shape[1]), device=source.device)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        weights = weights.unsqueeze(dim=-1)

        source_centroid = torch.sum(source * weights, dim=1)
        template_centroid = torch.sum(template * weights, dim=1)
        
        source_centered = source - source_centroid.unsqueeze(dim=1)
        template_centered = template - template_centroid.unsqueeze(dim=1)
          
        H = torch.matmul(source_centered.transpose(-1, -2), template_centered * weights)

        U, S, V = torch.svd(H, some=False) 
        R_pos = torch.matmul(V, U.transpose(-1, -2))
        V_neg = V.clone()
        V_neg[:, :, 2] *= -1
        R_neg = torch.matmul(V_neg, U.transpose(-1, -2))
        R = torch.where(torch.det(R_pos).unsqueeze(dim=-1).unsqueeze(dim=-1) > 0, R_pos, R_neg)

        if self.calculate_t:
            t = torch.matmul(-R, source_centroid.unsqueeze(dim=-1)) + template_centroid.unsqueeze(dim=-1)
            t = t.view(batch, 3)
        else:
            t = None
            
        R = R if self.calculate_R else None

        return R, t