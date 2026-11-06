# Copyright (c) SJTU. All rights reserved.
from copy import deepcopy

import torch
from mmdet.models.losses.utils import weighted_loss
from torch import nn

from ..builder import ROTATED_LOSSES


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def xy_stddev_pearson_2_xy_sigma(xy_stddev_pearson):
    """Convert oriented bounding box from the Pearson coordinate system to 2-D
    Gaussian distribution.

    Args:
        xy_stddev_pearson (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xy_stddev_pearson.shape
    assert _shape[-1] == 5
    xy = xy_stddev_pearson[..., :2]
    stddev = xy_stddev_pearson[..., 2:4]
    pearson = xy_stddev_pearson[..., 4].clamp(min=1e-7 - 1, max=1 - 1e-7)
    covar = pearson * stddev.prod(dim=-1)
    var = stddev.square()
    sigma = torch.stack((var[..., 0], covar, covar, var[..., 1]),
                        dim=-1).reshape(_shape[:-1] + (2, 2))
    return xy, sigma


def postprocess(distance, fun='log1p', tau=1.0):
    """Convert distance to loss.

    Args:
        distance (torch.Tensor)
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'sqrt':
        distance = torch.sqrt(distance.clamp(1e-7))
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance


@weighted_loss
def gwcd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sigma_angle=1.0):
    """Gaussian-Weighted Chamfer Distance loss.

    Args:
        pred (torch.Tensor): Predicted bboxes (xy_wh_r format, last dim 5)
        target (torch.Tensor): GT bboxes (xy_wh_r format, last dim 5)
        fun (str): Postprocess function for distance
        tau (float): Postprocess parameter
        alpha (float): Weight balance parameter
        sigma_angle (float): Sigma for angle Gaussian weight
    """
    # 确保pred和target是原始旋转框格式（5维）
    # 如果已经是元组格式（预处理后的），我们需要原始数据
    if isinstance(pred, tuple):
        # 对于新损失函数，我们需要原始旋转框数据
        # 这里我们无法直接获取原始数据，需要修改调用方式
        raise ValueError("gwcd_loss requires raw rotated boxes, not preprocessed tuples")

    # 检查维度
    if pred.dim() < 2 or pred.size(-1) != 5:
        raise ValueError(f"gwcd_loss expects pred with last dim 5, got {pred.shape}")
    if target.dim() < 2 or target.size(-1) != 5:
        raise ValueError(f"gwcd_loss expects target with last dim 5, got {target.shape}")

    # 解析预测/真实框的角度（最后一维为角度）
    angle_p = pred[..., 4]  # 预测角度
    angle_t = target[..., 4]  # 真实角度

    # 1. 计算高斯权重（基于角度距离）
    angle_diff = angle_p - angle_t
    angle_diff = torch.min(torch.abs(angle_diff), 2 * torch.pi - torch.abs(angle_diff))  # 周期修正
    weight_p = torch.exp(-(angle_diff ** 2) / (2 * sigma_angle ** 2))  # 预测点权重
    weight_t = torch.exp(-(angle_diff ** 2) / (2 * sigma_angle ** 2))  # 真实点权重（对称）

    # 2. 转换为高斯分布参数（复用现有函数）
    xy_p, sigma_p = xy_wh_r_2_xy_sigma(pred)
    xy_t, sigma_t = xy_wh_r_2_xy_sigma(target)

    # 3. 计算加权倒角距离（简化版：用中心点距离近似轮廓距离）
    dist_p2t = torch.norm(xy_p - xy_t, dim=-1)  # 预测中心到真实中心的距离
    dist_t2p = torch.norm(xy_t - xy_p, dim=-1)  # 真实中心到预测中心的距离

    # 加权求和（模拟轮廓点集加权）
    chamfer_dist = (weight_p * dist_p2t + weight_t * dist_t2p) / 2  # 对称化处理

    return postprocess(chamfer_dist, fun=fun, tau=tau)


@weighted_loss
def pckl_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True, penalty_coef=0.1):
    """Periodicity-Corrected KL Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes (xy_wh_r format, last dim 5)
        target (torch.Tensor): GT bboxes (xy_wh_r format, last dim 5)
        fun (str): Postprocess function for distance
        tau (float): Postprocess parameter
        alpha (float): Weight balance parameter
        sqrt (bool): Whether to sqrt the distance
        penalty_coef (float): Coefficient for periodic penalty term
    """
    # 确保pred和target是原始旋转框格式（5维）
    if isinstance(pred, tuple):
        raise ValueError("pckl_loss requires raw rotated boxes, not preprocessed tuples")

    # 检查维度
    if pred.dim() < 2 or pred.size(-1) != 5:
        raise ValueError(f"pckl_loss expects pred with last dim 5, got {pred.shape}")
    if target.dim() < 2 or target.size(-1) != 5:
        raise ValueError(f"pckl_loss expects target with last dim 5, got {target.shape}")

    # 转换为高斯分布参数（复用现有函数）
    xy_p, sigma_p = xy_wh_r_2_xy_sigma(pred)
    xy_t, sigma_t = xy_wh_r_2_xy_sigma(target)

    # 1. 角度周期性修正（针对均值差）
    angle_p = pred[..., 4]
    angle_t = target[..., 4]
    delta_mu = angle_p - angle_t
    # 周期修正：确保差值在[0, π]
    delta_mu_corr = torch.min(torch.abs(delta_mu), 2 * torch.pi - torch.abs(delta_mu))

    # 2. 基础KL散度计算（复用原有逻辑）
    _shape = xy_p.shape
    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    sigma_p = sigma_p.reshape(-1, 2, 2)
    sigma_t = sigma_t.reshape(-1, 2, 2)

    sigma_p_inv = torch.stack((sigma_p[..., 1, 1], -sigma_p[..., 0, 1],
                               -sigma_p[..., 1, 0], sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    sigma_p_inv = sigma_p_inv / sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(sigma_p_inv).bmm(dxy).view(-1)

    whr_distance = 0.5 * sigma_p_inv.bmm(sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    sigma_p_det_log = sigma_p.det().log()
    sigma_t_det_log = sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (sigma_p_det_log - sigma_t_det_log) - 1

    # 3. 注入角度周期修正项
    kl_base = (xy_distance / (alpha * alpha) + whr_distance).reshape(_shape[:-1])
    # 周期性惩罚项（修正大角度差的过度影响）
    k = torch.round(delta_mu / (2 * torch.pi))  # 周期调整系数
    penalty = penalty_coef * torch.exp(-((delta_mu - 2 * torch.pi * k) ** 2) / (2 * alpha ** 2))
    kl_corrected = kl_base - penalty  # 减去惩罚项（缩小偏差影响）

    if sqrt:
        kl_corrected = kl_corrected.clamp(1e-7).sqrt()

    return postprocess(kl_corrected, fun=fun, tau=tau)


@weighted_loss
def gwd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """Gaussian Wasserstein distance loss."""
    # 确保pred和target是元组格式（xy, sigma）
    if not isinstance(pred, tuple):
        pred = xy_wh_r_2_xy_sigma(pred)
    if not isinstance(target, tuple):
        target = xy_wh_r_2_xy_sigma(target)

    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    """Kullback-Leibler Divergence loss."""
    # 确保pred和target是元组格式（xy, sigma）
    if not isinstance(pred, tuple):
        pred = xy_wh_r_2_xy_sigma(pred)
    if not isinstance(target, tuple):
        target = xy_wh_r_2_xy_sigma(target)

    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(Sigma_t).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(1e-7).sqrt()

    distance = distance.reshape(_shape[:-1])

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def jd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    """Symmetrical Kullback-Leibler Divergence loss."""
    jd = kld_loss(
        pred,
        target,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=False,
        reduction='none')
    jd = jd + kld_loss(
        target,
        pred,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=False,
        reduction='none')
    jd = jd * 0.5
    if sqrt:
        jd = jd.clamp(1e-7).sqrt()
    return postprocess(jd, fun=fun, tau=tau)


@weighted_loss
def kld_symmax_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    """Symmetrical Max Kullback-Leibler Divergence loss."""
    kld_pt = kld_loss(
        pred,
        target,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=sqrt,
        reduction='none')
    kld_tp = kld_loss(
        target,
        pred,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=sqrt,
        reduction='none')
    kld_symmax = torch.max(kld_pt, kld_tp)
    return postprocess(kld_symmax, fun=fun, tau=tau)


@weighted_loss
def kld_symmin_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    """Symmetrical Min Kullback-Leibler Divergence loss."""
    kld_pt = kld_loss(
        pred,
        target,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=sqrt,
        reduction='none')
    kld_tp = kld_loss(
        target,
        pred,
        fun='none',
        tau=0,
        alpha=alpha,
        sqrt=sqrt,
        reduction='none')
    kld_symmin = torch.min(kld_pt, kld_tp)
    return postprocess(kld_symmin, fun=fun, tau=tau)


@ROTATED_LOSSES.register_module()
class GDLoss(nn.Module):
    """Gaussian based loss.

    Args:
        loss_type (str):  Type of loss.
        representation (str, optional): Coordinate System.
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        alpha (float, optional): Defaults to 1.0.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    BAG_GD_LOSS = {
        'gwd': gwd_loss,
        'kld': kld_loss,
        'jd': jd_loss,
        'kld_symmax': kld_symmax_loss,
        'kld_symmin': kld_symmin_loss,
        'gwcd': gwcd_loss,  # 新增：高斯加权倒角距离
        'pckl': pckl_loss  # 新增：周期性修正KL散度
    }

    # 需要原始旋转框的损失函数
    RAW_BOX_LOSSES = {'gwcd', 'pckl'}

    BAG_PREP = {
        'xy_stddev_pearson': xy_stddev_pearson_2_xy_sigma,
        'xy_wh_r': xy_wh_r_2_xy_sigma
    }

    def __init__(self,
                 loss_type,
                 representation='xy_wh_r',
                 fun='log1p',
                 tau=0.0,
                 alpha=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(GDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'none', 'sqrt']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.preprocess = self.BAG_PREP[representation]
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kwargs = kwargs
        self.requires_raw_boxes = loss_type in self.RAW_BOX_LOSSES

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        # 根据损失函数类型决定是否进行预处理
        if self.requires_raw_boxes:
            # 对于需要原始旋转框的损失函数，直接使用原始数据
            pred_processed = pred
            target_processed = target
        else:
            # 对于其他损失函数，进行预处理
            pred_processed = self.preprocess(pred)
            target_processed = self.preprocess(target)

        return self.loss(
            pred_processed,
            target_processed,
            fun=self.fun,
            tau=self.tau,
            alpha=self.alpha,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction,
            **_kwargs) * self.loss_weight