import torch
import torch.nn as nn
from mmrotate.models.roi_heads.bbox_heads import RotatedShared2FCBBoxHead
from mmrotate.models.builder import ROTATED_HEADS, build_backbone, build_neck, build_head


@ROTATED_HEADS.register_module()
class INRDBBoxHead(RotatedShared2FCBBoxHead):
    """
    适配 MMRotate 0.3.4 的 INRD BBoxHead
    【核心适配】：
    适配点：
    1. forward 回归原生签名（仅接收 x），rois/roi_extractor 通过实例属性传递
    2. 移除 BBoxHead 中非法的 self.roi_extractor 引用（RoI 提取器属于 RoIHead）
    3. 保持 IRFF→LAR-ISE→FFRD 的协同逻辑不变
    """

    def __init__(self,
                 irff_cfg=None,
                 lar_ise_cfg=None,
                 ffrd_cfg=None,
                 equiv_loss_weight=0.1,
                 **kwargs):
        # 强制设置 ignore_index，交给父类处理忽略标签
        if 'loss_cls' in kwargs and isinstance(kwargs['loss_cls'], dict):
            kwargs['loss_cls']['ignore_index'] = -1
        super(INRDBBoxHead, self).__init__(**kwargs)

        # 核心模块初始化
        self.irff = build_backbone(irff_cfg) if irff_cfg else None
        self.lar_ise = build_neck(lar_ise_cfg) if lar_ise_cfg else None
        self.ffrd = build_head(ffrd_cfg) if ffrd_cfg else None

        # 损失权重
        self.equiv_loss_weight = equiv_loss_weight
        # 标记是否启用FFRD
        self.use_ffrd_align = (self.ffrd is not None and self.irff is not None)

        # 动态属性：由 RoIHead 传入（适配原生调用）
        self.current_rois = None  # 存储当前批次的 rois
        self.current_roi_extractor = None  # 存储 RoIHead 的 roi_extractor
        self.equiv_loss = torch.tensor(0.0)  # 等变损失初始化

    def set_roi_info(self, rois=None, roi_extractor=None):
        """
        由 RoIHead 调用，传递 rois 和 roi_extractor（适配原生逻辑的关键）
        Args:
            rois (Tensor): [N, 6] 格式的 RoIs（batch_idx + 旋转框坐标）
            roi_extractor (nn.Module): RoIHead 中的 bbox_roi_extractor
        """
        self.current_rois = rois
        self.current_roi_extractor = roi_extractor
        # 重置等变损失
        if rois is not None:
            self.equiv_loss = torch.tensor(0.0, device=rois.device)
        else:
            self.equiv_loss = torch.tensor(0.0)

    def forward(self, x):
        """
        回归原生 forward 签名（仅接收 x），兼容 mmrotate 上游调用逻辑
        【核心保留】：
        Args:
            x (Tensor): 原生 RoI Align 提取的特征（或 backbone 特征，取决于 FFRD 是否启用）
        Returns:
            cls_score (Tensor): 分类得分
            bbox_pred (Tensor): 回归预测
        """
        rois = self.current_rois
        roi_extractor = self.current_roi_extractor
        device = x.device if isinstance(x, torch.Tensor) else x[0].device

        # 初始化等变损失（确保设备一致性）
        self.equiv_loss = self.equiv_loss.to(device)

        if self.use_ffrd_align and rois is not None and roi_extractor is not None:
            # 1. 提取候选框（剔除 batch_idx，保留旋转框坐标 [cx, cy, w, h, a]）
            bboxes = rois[:, 1:6]  # [N, 5] (x1,y1,x2,y2,theta) 适配你的坐标格式

            compute_equiv = self.training
            roi_feat, self.equiv_loss = self.ffrd(bboxes, self.irff, compute_equiv=compute_equiv)  # 新增参数

            # 方法1：通道注意力（FFRD生成权重，调制RoI特征）
            ffrd_weight = torch.sigmoid(roi_feat.mean(dim=[2, 3], keepdim=True))  # [N,256,1,1]
            roi_feat = x * ffrd_weight  # 用FFRD调制RoI特征

            # 方法2：残差融合
            #roi_feat = x + roi_feat * 0.1

            # 方法3：双路径融合：通道注意力 + 残差连接
            #ffrd_weight = torch.sigmoid(roi_feat.mean(dim=[2, 3], keepdim=True))
            #roi_feat = x * ffrd_weight + roi_feat * 0.15  # 增加残差路径，保留FFRD原始信息

            # 3. LAR-ISE 大长宽比形状增强（方案协同逻辑完全保留）
            if self.lar_ise is not None:
                # 计算候选框中心、角度、长宽比
                x_center = ((bboxes[:, 0] + bboxes[:, 2]) / 2.0).unsqueeze(-1)
                y_center = ((bboxes[:, 1] + bboxes[:, 3]) / 2.0).unsqueeze(-1)
                theta_center = bboxes[:, 4].unsqueeze(-1)
                w = bboxes[:, 2] - bboxes[:, 0]
                h = bboxes[:, 3] - bboxes[:, 1]
                ar = (w / (h + 1e-6)).clamp(1e-6, 1e6).unsqueeze(-1)

                # LAR-ISE 形状感知特征增强
                lar_feat = self.lar_ise(x_center, y_center, theta_center, ar, bboxes, self.irff)
                lar_feat = lar_feat.view(bboxes.shape[0], self.in_channels, 1, 1).expand(-1, -1, 7, 7)
                roi_feat = roi_feat + lar_feat  # 特征融合

            # 4. FFRD 已直接输出 [N, C, 7, 7] 空间特征，无需重复
            # 确保数值稳定
            roi_feat = roi_feat.clamp(-1e3, 1e3)

        # 降级逻辑：未启用 FFRD 时，使用原生 RoI Align 特征（x 已是 RoI 特征）
        else:
            roi_feat = x
        # ==============================================================================

        # 特征数值稳定处理（保留）
        roi_feat = roi_feat.clamp(-1e3, 1e3)

        # 调用父类原生 forward，完成分类+回归（保留）
        cls_score, bbox_pred = super(INRDBBoxHead, self).forward(roi_feat)

        # 优化：加判断后清空，避免None赋值异常
        if self.current_rois is not None:
            self.current_rois = None
        if self.current_roi_extractor is not None:
            self.current_roi_extractor = None

        return cls_score, bbox_pred

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights,
             reduction_override=None):
        """
        损失计算：原生分类/回归损失 + IRFF 旋转等变损失（逻辑完全保留）
        """
        # 标签类型转换，解决 Float 类型报错（保留）
        labels = labels.long() if labels is not None else None

        # 原生分类+回归损失（保留）
        losses = super(INRDBBoxHead, self).loss(
            cls_score, bbox_pred, rois, labels, label_weights,
            bbox_targets, bbox_weights, reduction_override
        )

        # 新增 IRFF 旋转等变约束损失（保留）
        if self.training and not torch.isnan(self.equiv_loss) and not torch.isinf(self.equiv_loss):
            losses['loss_equiv'] = self.equiv_loss_weight * self.equiv_loss

        return losses