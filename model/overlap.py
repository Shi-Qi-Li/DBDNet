from typing import Dict

import torch
import torch.nn as nn

from .dgcnn import EdgeConv
from .basic_block import Conv1dModule, TranslationDown, TranslationUp, QueryDown, Self_Cross_Attention
from .position_embedding import PositionEmbeddingCoordsSine
from .builder import MODEL
    
class HRNetEncoder(nn.Module):
    def __init__(self, in_channel: int = 3, k: int = 20, base_channel: int = 32):
        super(HRNetEncoder, self).__init__()
        self.base_channel = base_channel

        self.stage1_conv1 = EdgeConv(in_channel, base_channel, k, base_channel, dict(type = "gn", num_groups = 4, num_channels = base_channel))
        self.stage1_down_1_2 = TranslationDown(k, base_channel, base_channel * 2)

        self.stage2_conv1 = EdgeConv(base_channel, base_channel, k, base_channel, dict(type = "gn", num_groups = 4, num_channels = base_channel))
        self.stage2_conv2 = EdgeConv(base_channel * 2, base_channel * 2, k, base_channel * 2, dict(type = "gn", num_groups = 4, num_channels = base_channel * 2))
        self.stage2_up_2_1 = TranslationUp(base_channel * 2, base_channel)
        self.stage2_down_1_2 = QueryDown(base_channel, base_channel * 2)
        self.stage2_down_1_3 = QueryDown(base_channel, base_channel * 4)
        self.stage2_down_2_3 = TranslationDown(k, base_channel * 2, base_channel * 4)

        self.stage3_conv1 = EdgeConv(base_channel, base_channel, k, base_channel, dict(type = "gn", num_groups = 4, num_channels = base_channel))
        self.stage3_conv2 = EdgeConv(base_channel * 2, base_channel * 2, k, base_channel * 2, dict(type = "gn", num_groups = 4, num_channels = base_channel * 2))
        self.stage3_conv3 = EdgeConv(base_channel * 4, base_channel * 4, k, base_channel * 4, dict(type = "gn", num_groups = 8, num_channels = base_channel * 4))
        self.stage3_up_2_1 = TranslationUp(base_channel * 2, base_channel)
        self.stage3_up_3_1 = TranslationUp(base_channel * 4, base_channel)
        self.stage3_up_3_2 = TranslationUp(base_channel * 4, base_channel * 2)        
        self.stage3_down_1_2 = QueryDown(base_channel, base_channel * 2)
        self.stage3_down_1_3 = QueryDown(base_channel, base_channel * 4)
        self.stage3_down_1_4 = QueryDown(base_channel, base_channel * 8)
        self.stage3_down_2_3 = QueryDown(base_channel * 2, base_channel * 4)
        self.stage3_down_2_4 = QueryDown(base_channel * 2, base_channel * 8)
        self.stage3_down_3_4 = TranslationDown(k, base_channel * 4, base_channel * 8)

        self.stage4_conv1 = EdgeConv(base_channel, base_channel, k, base_channel, dict(type = "gn", num_groups = 4, num_channels = base_channel))
        self.stage4_conv2 = EdgeConv(base_channel * 2, base_channel * 2, k, base_channel * 2, dict(type = "gn", num_groups = 4, num_channels = base_channel * 2))
        self.stage4_conv3 = EdgeConv(base_channel * 4, base_channel * 4, k, base_channel * 4, dict(type = "gn", num_groups = 8, num_channels = base_channel * 4))
        self.stage4_conv4 = EdgeConv(base_channel * 8, base_channel * 8, k, base_channel * 8, dict(type = "gn", num_groups = 8, num_channels = base_channel * 8))
        
        self.stage4_up_2_1 = TranslationUp(base_channel * 2, base_channel)
        self.stage4_up_3_1 = TranslationUp(base_channel * 4, base_channel)
        self.stage4_up_4_1 = TranslationUp(base_channel * 8, base_channel)
        self.stage4_up_3_2 = TranslationUp(base_channel * 4, base_channel * 2)        
        self.stage4_up_4_2 = TranslationUp(base_channel * 8, base_channel * 2)
        self.stage4_up_4_3 = TranslationUp(base_channel * 8, base_channel * 4)
        self.stage4_down_1_2 = QueryDown(base_channel, base_channel * 2)
        self.stage4_down_1_3 = QueryDown(base_channel, base_channel * 4)
        self.stage4_down_1_4 = QueryDown(base_channel, base_channel * 8)
        self.stage4_down_2_3 = QueryDown(base_channel * 2, base_channel * 4)
        self.stage4_down_2_4 = QueryDown(base_channel * 2, base_channel * 8)
        self.stage4_down_3_4 = QueryDown(base_channel * 4, base_channel * 8)
        
        self.fusion_1 = Conv1dModule(base_channel * 5, base_channel)
        self.fusion_2 = Conv1dModule(base_channel * 2 * 4, base_channel * 2)
        self.fusion_3 = Conv1dModule(base_channel * 4 * 3, base_channel * 4)
        self.fusion_4 = Conv1dModule(base_channel * 8 * 2, base_channel * 8)
        
        self.neck_2 = TranslationUp(base_channel * 2, base_channel)
        self.neck_3 = TranslationUp(base_channel * 4, base_channel)
        self.neck_4 = TranslationUp(base_channel * 8, base_channel) 

    def forward(self, pc: torch.Tensor):
        # Stage1
        feature_s1_1 = self.stage1_conv1(pc)
        # Trans1
        pc, feature_s2_1 = pc, feature_s1_1
        pc_2, feature_s2_2, indices_2 = self.stage1_down_1_2(pc, feature_s1_1)

        # Stage2
        feature_s2_1 = self.stage2_conv1(feature_s2_1)
        feature_s2_2 = self.stage2_conv2(feature_s2_2)
        # Trans2
        pc, feature_s3_1 = pc, feature_s2_1 + self.stage2_up_2_1(pc_2, pc, feature_s2_2)
        pc_2, feature_s3_2 = pc_2, feature_s2_2 + self.stage2_down_1_2(feature_s2_1, indices_2)
        pc_3, feature_s3_3, indices_3 = self.stage2_down_2_3(pc_2, feature_s2_2)
        feature_s3_3 += self.stage2_down_1_3(feature_s2_1, [indices_2, indices_3])

        #stage3
        feature_s3_1 = self.stage3_conv1(feature_s3_1)
        feature_s3_2 = self.stage3_conv2(feature_s3_2)
        feature_s3_3 = self.stage3_conv3(feature_s3_3)
        # Trans3
        pc, feature_s4_1 = pc, feature_s3_1 + self.stage3_up_2_1(pc_2, pc, feature_s3_2) + self.stage3_up_3_1(pc_3, pc, feature_s3_3)
        pc_2, feature_s4_2 = pc_2, feature_s3_2 + self.stage3_down_1_2(feature_s3_1, indices_2) + self.stage3_up_3_2(pc_3, pc_2, feature_s3_3)
        pc_3, feature_s4_3 = pc_3, feature_s3_3 + self.stage3_down_1_3(feature_s3_1, [indices_2, indices_3]) + self.stage3_down_2_3(feature_s3_2, indices_3)
        pc_4, feature_s4_4, indices_4 = self.stage3_down_3_4(pc_3, feature_s3_3)
        feature_s4_4 += self.stage3_down_1_4(feature_s3_1, [indices_2, indices_3, indices_4]) + self.stage3_down_2_4(feature_s3_2, [indices_3, indices_4])

        #stage4
        feature_s4_1 = self.stage4_conv1(feature_s4_1)
        feature_s4_2 = self.stage4_conv2(feature_s4_2)
        feature_s4_3 = self.stage4_conv3(feature_s4_3)
        feature_s4_4 = self.stage4_conv4(feature_s4_4)
        # Trans4
        pc, feature_s5_1 = pc, feature_s4_1 + self.stage4_up_2_1(pc_2, pc, feature_s4_2) + self.stage4_up_3_1(pc_3, pc, feature_s4_3) + self.stage4_up_4_1(pc_4, pc, feature_s4_4)
        pc_2, feature_s5_2 = pc_2, feature_s4_2 + self.stage4_down_1_2(feature_s4_1, indices_2) + self.stage4_up_3_2(pc_3, pc_2, feature_s4_3) + self.stage4_up_4_2(pc_4, pc_2, feature_s4_4)
        pc_3, feature_s5_3 = pc_3, feature_s4_3 + self.stage4_down_1_3(feature_s4_1, [indices_2, indices_3]) + self.stage4_down_2_3(feature_s4_2, indices_3) + self.stage4_up_4_3(pc_4, pc_3, feature_s4_4)
        pc_4, feature_s5_4 = pc_4, feature_s4_4 + self.stage4_down_1_4(feature_s4_1, [indices_2, indices_3, indices_4]) + self.stage4_down_2_4(feature_s4_2, [indices_3, indices_4]) + self.stage4_down_3_4(feature_s4_3, indices_4)

        feature_1 = self.fusion_1(torch.cat([feature_s1_1, feature_s2_1, feature_s3_1, feature_s4_1, feature_s5_1], dim=1))
        feature_2 = self.fusion_2(torch.cat([feature_s2_2, feature_s3_2, feature_s4_2, feature_s5_2], dim=1))
        feature_3 = self.fusion_3(torch.cat([feature_s3_3, feature_s4_3, feature_s5_3], dim=1))
        feature_4 = self.fusion_4(torch.cat([feature_s4_4, feature_s5_4], dim=1))
        
        return torch.cat([feature_1, self.neck_2(pc_2, pc, feature_2), self.neck_3(pc_3, pc, feature_3), self.neck_4(pc_4, pc, feature_4)], dim=1)

    @property
    def embedding_dim(self) -> int:
        return self.base_channel * 4

@MODEL
class HROverlapNet(nn.Module):
    def __init__(self, in_channel: int = 3, k: int = 20, base_channel: int = 32, layer_num: int = 2):
        super(HROverlapNet, self).__init__()
        self.in_channel = in_channel

        self.encoder = HRNetEncoder(in_channel, k, base_channel)
        
        self.pe = PositionEmbeddingCoordsSine(d_model=self.encoder.embedding_dim)
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.encoder.embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.self_layer_norm = nn.LayerNorm(self.encoder.embedding_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.encoder.embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.cross_layer_norm = nn.LayerNorm(self.encoder.embedding_dim)
        
        self.attention_layers = []
        for _ in range(layer_num):
            self.attention_layers.append(Self_Cross_Attention(self.encoder.embedding_dim))

        self.attention = nn.Sequential(*self.attention_layers)

        self.decoder = nn.Sequential(
            Conv1dModule(self.encoder.embedding_dim * 2, 512, dict(type = "gn", num_groups = 8, num_channels = 512)),
            Conv1dModule(512, 256, dict(type = "gn", num_groups = 8, num_channels = 256)),
            Conv1dModule(256, 128, dict(type = "gn", num_groups = 4, num_channels = 128)),
        )

        self.linear = nn.Sequential(
			nn.Conv1d(128, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source, template = data["source_pc"], data["template_pc"]
        B, D, N = source.shape

        source_feature, template_feature = self.encoder(source), self.encoder(template)

        source_w_pe = source_feature.permute(0, 2, 1) + self.pe(source.permute(0, 2, 1))
        template_w_pe = template_feature.permute(0, 2, 1) + self.pe(template.permute(0, 2, 1))
        
        source_feature, template_feature = self.attention((source_w_pe, template_w_pe))

        source_feature = source_feature.permute(0, 2, 1)
        template_feature = template_feature.permute(0, 2, 1)

        source_max = source_feature.max(dim=-1, keepdim=True)[0]
        template_max = template_feature.max(dim=-1, keepdim=True)[0]
        

        source_feature = torch.cat([source_feature, template_max.repeat(1, 1, N)], dim=1)
        template_feature = torch.cat([template_feature, source_max.repeat(1, 1, N)], dim=1)

        source_feature = self.decoder(source_feature)
        template_feature = self.decoder(template_feature)
        
        mask_s = self.linear(source_feature)
        mask_t = self.linear(template_feature)

        mask_s = mask_s.view(B, -1)
        mask_t = mask_t.view(B, -1)

        predictions = {
            "template_mask_pred": [mask_t],
            "source_mask_pred": [mask_s]
        }
        
        return predictions

    def create_input(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "source_pc": data_batch["source_pc"].permute(0, 2, 1).contiguous(),
            "template_pc": data_batch["template_pc"].permute(0, 2, 1).contiguous()
        }
    
    def create_ground_truth(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "source_mask_gt": data_batch["source_mask"],
            "template_mask_gt": data_batch["template_mask"]
        }
