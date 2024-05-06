from typing import List, Union, Optional

import torch
import torch.nn as nn

from .basic_block import Conv2dModule


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    B, _, N = x.shape
    x = x.view(B, -1, N)
    device = x.device
    if idx is None:
        idx = knn(x, k)
    
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    D = x.shape[1]

    x = x.transpose(2, 1).contiguous()
    feat = x.view(B * N, -1)[idx, :]
    feat = feat.view(B, N, k, D)
    x = x.view(B, N, 1, D).repeat(1, 1, k, 1)

    feat = torch.cat([feat - x, x], dim=3).permute(0, 3, 1, 2).contiguous()

    return feat # (batch_size, D, num_points, k)

class EdgeConv(nn.Module):
    def __init__(
        self, 
        in_channel: int, 
        out_channel: int, 
        k: int = 20, 
        hidden_channel: Optional[Union[int, List]] = None, 
        norm_cfg: dict = dict(type = None),
        act_cfg: dict = dict(type = "LeakyReLu", negative_slope = 0.2, inplace = True)
    ):
        super(EdgeConv, self).__init__()
        self.k = k
        if isinstance(hidden_channel, int):
            hidden_channel = [hidden_channel]

        self.conv_layers = []
        in_channel = in_channel * 2
        if hidden_channel:
            for channel in hidden_channel:
                self.conv_layers.append(Conv2dModule(in_channel, channel, norm_cfg, act_cfg))
                in_channel = channel
        self.conv_layers.append(Conv2dModule(in_channel, out_channel, norm_cfg, act_cfg))

        self.conv = nn.Sequential(*self.conv_layers)

    def forward(self, x):
        x = get_graph_feature(x, self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x