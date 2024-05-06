from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_block import Self_Cross_Attention
from .optimal_transport import sinkhorn
from .overlap import HRNetEncoder, HROverlapNet
from .svd import SVDHead
from .position_embedding import PositionEmbeddingCoordsSine
from .builder import MODEL

from utils import batch_transform, batch_get_points_by_index, batch_square_distance, batch_get_overlap_index


class ParameterPredictionNet(nn.Module):
    def __init__(self) -> None:
        super(ParameterPredictionNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(inplace=True),
        )
        
        self.pre_parameter_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 4)
        )

    def forward(self, source, template):
        pad = (0, 0, 0, 1)
        source = F.pad(source, pad, mode='constant', value=0)
        template = F.pad(template, pad, mode='constant', value=1)
        
        feature = self.encoder(torch.cat([source, template], dim=-1))
                
        global_feature = torch.max(feature, dim=-1)[0]
        pre_weights = self.pre_parameter_layer(global_feature)

        beta1 = F.softplus(pre_weights[:, 0])
        alpha1 = F.softplus(pre_weights[:, 1])
        beta2 = F.softplus(pre_weights[:, 2])
        alpha2 = F.softplus(pre_weights[:, 3])

        return beta1, alpha1, beta2, alpha2

def compute_affinity(beta: torch.Tensor, feat_distance: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if beta.ndim == 1 and alpha.ndim == 1:
        hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
    elif beta.ndim == 2 and alpha.ndim == 2:
        hybrid_affinity = beta[:, :, None] * (feat_distance - alpha[:, :, None])
    return hybrid_affinity

class RegNet(nn.Module):
    def __init__(self, embedding_dim: int = 256, layer_num: int = 2, keep_ratio: float = 1.0):
        super(RegNet, self).__init__() 
        self.embedding_dim = embedding_dim
        self.keep_ratio = keep_ratio

        self.weight_net = ParameterPredictionNet()
        
        self.pe = PositionEmbeddingCoordsSine(
            d_model=embedding_dim
        )

        self.attention_layers = []
        self.attention_R_layers = []
        self.attention_t_layers = []
        for _ in range(layer_num // 2):
            self.attention_layers.append(Self_Cross_Attention(embedding_dim))
            self.attention_R_layers.append(Self_Cross_Attention(embedding_dim))
            self.attention_t_layers.append(Self_Cross_Attention(embedding_dim))

        self.attention = nn.Sequential(*self.attention_layers)
        self.attention_R = nn.Sequential(*self.attention_R_layers)
        self.attention_t = nn.Sequential(*self.attention_t_layers)

        self.svd = SVDHead(keep_ratio=keep_ratio)
    
    def forward(self, source: torch.Tensor, template: torch.Tensor, source_feature: torch.Tensor, template_feature: torch.Tensor):
        
        B, D, N = source.shape
        M = template.shape[-1]

        source_w_pe = source_feature.permute(0, 2, 1) + self.pe(source.permute(0, 2, 1))
        template_w_pe = template_feature.permute(0, 2, 1) + self.pe(template.permute(0, 2, 1))

        source_feature, template_feature = self.attention((source_w_pe, template_w_pe))

        source_feature_R, template_feature_R = source_feature, template_feature
        source_feature_t, template_feature_t = source_feature.clone(), template_feature.clone()

        source_feature_R, template_feature_R = self.attention_R((source_feature_R, template_feature_R))
        source_feature_t, template_feature_t = self.attention_t((source_feature_t, template_feature_t))

        beta1, alpha1, beta2, alpha2 = self.weight_net(source, template)
        score_R = batch_square_distance(source_feature_R, template_feature_R)
        score_t = batch_square_distance(source_feature_t, template_feature_t)

        score_R = compute_affinity(beta1, score_R, alpha=alpha1)
        score_t = compute_affinity(beta2, score_t, alpha=alpha2)

        score = torch.cat([score_R.unsqueeze(dim=1), score_t.unsqueeze(dim=1)], dim=1)
        
        log_permutation_matrix = sinkhorn(score, 5, True)
        permutation_matrix = torch.exp(log_permutation_matrix)
        
        permutated_template_R = torch.bmm(permutation_matrix[:, 0, :, :], template.permute(0, 2, 1))
        permutated_template_t = torch.bmm(permutation_matrix[:, 1, :, :], template.permute(0, 2, 1))
      
        if self.keep_ratio < 1 and self.training:
            batch_R_1, _ = self.svd(source.permute(0, 2, 1), permutated_template_R)
            _, batch_t_1 = self.svd(source.permute(0, 2, 1), permutated_template_t)

            batch_R_2, _ = self.svd(source.permute(0, 2, 1), permutated_template_R)
            _, batch_t_2 = self.svd(source.permute(0, 2, 1), permutated_template_t)

            batch_R = (batch_R_1, batch_R_2)
            batch_t = (batch_t_1, batch_t_2)
        else:
            batch_R, _ = self.svd(source.permute(0, 2, 1), permutated_template_R)
            _, batch_t = self.svd(source.permute(0, 2, 1), permutated_template_t)
        
        weight = None
        h = None
        
        return batch_R, batch_t, permutation_matrix, weight, h
    
@MODEL
class DBDNet(nn.Module):
    def __init__(self, in_channel: int = 3, k: int = 20, base_channel: int = 64, overlap_channel: int = 64, layer_num: int = 2, overlap_layer_num: int = 2, train_iteration: int = 1, test_iteration: int = 4, keep_ratio: float = 1.0, share_weights: bool = True):
        super(DBDNet, self).__init__()
        self.in_channel = in_channel
        self.train_iteration = train_iteration
        self.test_iteration = test_iteration
        self.share_weights = share_weights

        self.overlap_predictor = HROverlapNet(in_channel, k, overlap_channel, overlap_layer_num)
        for parameter in self.overlap_predictor.parameters():
            parameter.requires_grad_(False)

        self.feature_extractor = HRNetEncoder(in_channel, k, base_channel)

        if share_weights:
            self.RegNet = RegNet(self.feature_extractor.embedding_dim, layer_num, keep_ratio)
        else:
            self.RegNet = nn.ModuleList([RegNet(self.feature_extractor.embedding_dim, keep_ratio) for _ in range(train_iteration)])

    def forward(self, data: Dict[str, torch.Tensor]):
        transformed_sources = []
        correspondences = []
        hs = []

        source, template = data["source_pc"], data["template_pc"]

        B = source.size()[0]
        
        mask = self.overlap_predictor(data)
        source_mask = mask["source_mask_pred"][0].view(B, -1)
        template_mask = mask["template_mask_pred"][0].view(B, -1)

        if B > 1:
            source_idx = batch_get_overlap_index(source_mask > 0.5)
            template_idx = batch_get_overlap_index(template_mask > 0.5)
        else:
            assert B == 1, "Invalid batch size"
            source_idx = torch.nonzero(source_mask[0] > 0.5).reshape(B, -1)
            if source_idx.shape[-1] < 160:
                _, source_idx = torch.topk(source_mask, 160, dim=1, sorted=True)
                
            template_idx = torch.nonzero(template_mask[0] > 0.5).reshape(B, -1)
            if template_idx.shape[-1] < 160:
                _, template_idx = torch.topk(template_mask, 160, dim=1, sorted=True)

        source = batch_get_points_by_index(source.permute(0, 2, 1), source_idx).permute(0, 2, 1)
        template = batch_get_points_by_index(template.permute(0, 2, 1), template_idx).permute(0, 2, 1)

        transformed_source = torch.clone(source)
        batch_R_res = torch.eye(3).to(source.device).unsqueeze(0).repeat(B, 1, 1)
        batch_t_res = torch.zeros(3, 1).to(source.device).unsqueeze(0).repeat(B, 1, 1)
        
        source_feature = self.feature_extractor(source)
        template_feature = self.feature_extractor(template)
        
        iteration = self.train_iteration if self.training else self.test_iteration
        for i in range(iteration):
            if self.share_weights:
                batch_R, batch_t, correspondence, weight, h = self.RegNet(transformed_source, template, source_feature, template_feature)
            else:
                batch_R, batch_t, correspondence, weight, h = self.RegNet[i](transformed_source, template, source_feature, template_feature)    
                
            if self.in_channel == 6:
                if isinstance(batch_R, torch.Tensor):    
                    transformed_points = batch_transform(transformed_source.permute(0, 2, 1)[:, :, :3].contiguous(), batch_R, batch_t)
                    transformed_normals = batch_transform(transformed_source.permute(0, 2, 1)[:, :, 3:].contiguous(), batch_R)
                elif isinstance(batch_R, Tuple):
                    transformed_points = batch_transform(transformed_source.permute(0, 2, 1)[:, :, :3].contiguous(), batch_R[0], batch_t[0])
                    transformed_normals = batch_transform(transformed_source.permute(0, 2, 1)[:, :, 3:].contiguous(), batch_R[0])
                else:
                    raise TypeError
                transformed_source = torch.cat([transformed_points, transformed_normals], dim=-1)
            else:
                if isinstance(batch_R, torch.Tensor):
                    transformed_source = batch_transform(transformed_source.permute(0, 2, 1).contiguous(), batch_R, batch_t)
                elif isinstance(batch_R, Tuple):
                    transformed_source = batch_transform(transformed_source.permute(0, 2, 1).contiguous(), batch_R[0], batch_t[0])
                else:
                    raise TypeError
            
            transformed_sources.append(transformed_source)
            hs.append(h)
            
            correspondences.append((correspondence[:, 0, :, :], correspondence[:, 1, :, :]))
            
            if isinstance(batch_R, torch.Tensor):
                batch_R_res = torch.bmm(batch_R, batch_R_res)
                batch_t_res = torch.bmm(batch_R, batch_t_res) + torch.unsqueeze(batch_t, -1)
            else:
                batch_R_res = torch.bmm(batch_R[0], batch_R_res)
                batch_t_res = torch.bmm(batch_R[0], batch_t_res) + torch.unsqueeze(batch_t[0], -1)
            transformed_source = transformed_source.permute(0, 2, 1).contiguous()
        batch_t_res = torch.squeeze(batch_t_res, dim=-1)
        
        predictions = {
            "R_pred": batch_R,
            "t_pred": batch_t,
            "pred_template_pcs": transformed_sources,
            "correspondences": correspondences,
            "template_mask_pred": [template_mask],
            "source_mask_pred": [source_mask],
            "source_idx_pred": source_idx,
            "template_idx_pred": template_idx
        }
        
        return predictions
    
    def create_input(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "source_pc": data_batch["source_pc"].permute(0, 2, 1).contiguous(),
            "template_pc": data_batch["template_pc"].permute(0, 2, 1).contiguous()
        }
    
    def create_ground_truth(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "R_gt": data_batch["R"],
            "t_gt": data_batch["t"],
            "source_mask_gt": data_batch["source_mask"],
            "template_mask_gt": data_batch["template_mask"],
            "source_pc": data_batch["source_pc"],
            "template_pc": data_batch["template_pc"]
        }