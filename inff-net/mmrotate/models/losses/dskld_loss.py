import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from ..builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weighted_loss


def von_mises_kld(theta_pred, theta_gt, kappa=10.0):
    """极坐标θ通道的Von-Mises分布KLD损失"""
    # theta_pred: (N, 1) 预测角度（弧度）
    # theta_gt: (N, 1) 真实角度（弧度）
    diff = theta_pred - theta_gt
    diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi  # 归一化到[-π, π]
    return -torch.log(torch.exp(kappa * torch.cos(diff)) / (2 * torch.pi * torch.i0(kappa)))


@ROTATED_LOSSES.register_module()
class DSKLD(nn.Module):
    """双频谱KLD损失：空间域KLD + 极坐标θ通道Von-Mises KLD"""

    def __init__(self,
                 spatial_weight=1.0,
                 angular_weight=1.0,
                 kappa=10.0,
                 reduction='mean'):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.angular_weight = angular_weight
        self.kappa = kappa
        self.reduction = reduction
        self.fp16_enabled = False

    @auto_fp16
    def forward(self,
                bbox_pred,  # (N, 5) [x, y, w, h, a]
                bbox_gt,  # (N, 5) [x, y, w, h, a]
                weight=None,
                avg_factor=None):
        # 1. 空间域KLD损失（针对x, y, w, h）
        mu_x_pred = bbox_pred[:, 0]
        mu_y_pred = bbox_pred[:, 1]
        sigma_x_pred = torch.exp(bbox_pred[:, 2] / 2)  # w -> sigma_x
        sigma_y_pred = torch.exp(bbox_pred[:, 3] / 2)  # h -> sigma_y

        mu_x_gt = bbox_gt[:, 0]
        mu_y_gt = bbox_gt[:, 1]
        sigma_x_gt = torch.exp(bbox_gt[:, 2] / 2)
        sigma_y_gt = torch.exp(bbox_gt[:, 3] / 2)

        spatial_kld = 0.5 * (
                (sigma_x_pred ** 2 / sigma_x_gt ** 2 + sigma_y_pred ** 2 / sigma_y_gt ** 2) +
                ((mu_x_gt - mu_x_pred) ** 2 / sigma_x_gt ** 2 + (mu_y_gt - mu_y_pred) ** 2 / sigma_y_gt ** 2) -
                2 + torch.log(sigma_x_gt ** 2 * sigma_y_gt ** 2 / (sigma_x_pred ** 2 * sigma_y_pred ** 2 + 1e-8))
        )

        # 2. 极坐标θ通道Von-Mises KLD损失
        theta_pred = bbox_pred[:, 4]
        theta_gt = bbox_gt[:, 4]
        angular_kld = von_mises_kld(theta_pred.unsqueeze(1), theta_gt.unsqueeze(1), self.kappa)

        # 总损失
        loss = self.spatial_weight * spatial_kld + self.angular_weight * angular_kld

        if weight is not None:
            loss = loss * weight
        if self.reduction == 'mean' and avg_factor is not None:
            loss = loss.sum() / avg_factor
        return loss