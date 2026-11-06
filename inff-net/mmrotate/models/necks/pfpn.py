import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmrotate.models.builder import ROTATED_NECKS
from mmdet.models.necks.fpn import FPN
import numpy as np
from mmcv.runner import BaseModule



class PolarMapping(BaseModule):
    """笛卡尔特征图 -> 极坐标特征图"""

    def __init__(self, angular_bins=16):
        super().__init__()
        self.angular_bins = angular_bins  # θ方向 bins

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        device = x.device

        # 生成网格坐标
        y = torch.linspace(-1, 1, H, device=device)
        x_grid = torch.linspace(-1, 1, W, device=device)
        xx, yy = torch.meshgrid(x_grid, y, indexing='xy')

        # 计算极坐标θ（[-π, π]）
        theta = torch.atan2(yy, xx)

        # 极坐标特征图展开（仅保留θ维度聚合）
        polar_feat = torch.zeros(B, C, self.angular_bins, device=device)
        for b in range(B):
            for c in range(C):
                # 按θ分bin池化
                theta_norm = (theta / torch.pi + 1) / 2 * (self.angular_bins - 1)  # [0, angular_bins-1]
                theta_indices = theta_norm.long().clamp(0, self.angular_bins - 1)
                polar_feat[b, c] = torch.stack([
                    x[b, c][theta_indices == i].mean() for i in range(self.angular_bins)
                ])
        return polar_feat  # (B, C, angular_bins)


@ROTATED_NECKS.register_module()
class PolarFPN(BaseModule):
    """极坐标特征金字塔：融合笛卡尔多尺度与极坐标θ条带特征"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 angular_bins=16,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.angular_bins = angular_bins

        if end_level == -1:
            self.backbone_end_level = len(in_channels)
            assert num_outs >= self.backbone_end_level - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.start_level = start_level
        self.end_level = end_level

        # 笛卡尔特征卷积
        self.lateral_convs = nn.ModuleList()
        # 极坐标特征映射
        self.polar_mappings = nn.ModuleList()
        # 极坐标特征卷积
        self.polar_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # 笛卡尔侧向卷积
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False)
            # 极坐标映射
            p_map = PolarMapping(angular_bins)
            # 极坐标特征卷积
            p_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=(1, angular_bins),  # θ方向卷积
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.polar_mappings.append(p_map)
            self.polar_convs.append(p_conv)

        # 输出卷积
        self.fpn_convs = nn.ModuleList()
        for i in range(num_outs):
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=activation,
                inplace=False)
            self.fpn_convs.append(fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        # inputs: 笛卡尔多尺度特征 (P3, P4, P5, ...)
        assert len(inputs) == len(self.in_channels)

        # 1. 笛卡尔特征侧向连接
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 2. 极坐标特征映射与卷积
        polar_feats = []
        for i, (lat, p_map, p_conv) in enumerate(zip(laterals, self.polar_mappings, self.polar_convs)):
            B, C, H, W = lat.shape
            # 极坐标映射 (B, C, angular_bins)
            polar_feat = p_map(lat)
            # 扩展维度后卷积 (B, C, 1, angular_bins) -> (B, C, 1, 1)
            polar_feat = p_conv(polar_feat.unsqueeze(2)).squeeze(2).squeeze(2)
            # 广播回笛卡尔特征尺寸 (B, C, H, W)
            polar_feat = polar_feat.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            polar_feats.append(polar_feat)

        # 3. 融合笛卡尔与极坐标特征
        fused_feats = [l + p for l, p in zip(laterals, polar_feats)]

        # 4. 自上而下融合（与标准FPN一致）
        used_backbone_levels = len(fused_feats)
        for i in range(used_backbone_levels - 1, 0, -1):
            fused_feats[i - 1] += F.interpolate(
                fused_feats[i],
                size=fused_feats[i - 1].shape[2:],
                mode='bilinear',
                align_corners=False)

        # 5. 输出卷积
        outs = [
            self.fpn_convs[i](fused_feats[i]) for i in range(used_backbone_levels)
        ]

        # 补全输出层级（如需要P6、P7）
        if self.num_outs > used_backbone_levels:
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return tuple(outs)