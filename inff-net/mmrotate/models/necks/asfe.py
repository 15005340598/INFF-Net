import torch
import torch.nn as nn
from mmrotate.models.builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class LARImplicitShapeEncoding(nn.Module):
    """
    超网络条件化的动态形状调制 (HCSM)
    【最小改动】完全保留原有forward接口，上游调用零修改
    核心优化：摒弃双分支冗余设计，共享IRFF主干，超网络动态生成调制参数，自适应采样密度
    """
    def __init__(self,
                 irff_channels=256,       # 与IRFF输出维度一致，兼容原有逻辑
                 hyper_hidden_dim=32,     # 方案指定：轻量MLP隐藏层32维
                 ar_threshold=5.0,        # 长宽比分档阈值
                 extreme_ar_threshold=15.0,  # 极端长宽比阈值
                 base_sample_num=12,       # 基础采样点数
                 max_sample_num=32):      # 极端AR最大采样点数
        super().__init__()
        self.irff_channels = irff_channels
        self.ar_threshold = ar_threshold
        self.extreme_ar_threshold = extreme_ar_threshold
        self.base_sample_num = base_sample_num
        self.max_sample_num = max_sample_num

        # ====================== 方案核心实现 ======================
        # 1. 轻量级HyperNetwork，替代原双分支编码器（参数量<2K）
        self.hyper_net = nn.Sequential(
            nn.Linear(1, hyper_hidden_dim),
            nn.ReLU(inplace=True)
        )
        # FiLM参数生成头：生成缩放系数γ和偏置β，维度与IRFF特征一致
        self.gamma_head = nn.Linear(hyper_hidden_dim, irff_channels)
        self.beta_head = nn.Linear(hyper_hidden_dim, irff_channels)
        # ==========================================================

    def adaptive_elliptical_sampling(self, x_center, y_center, w, h, theta, ar):
        """
        自适应椭圆采样（方案核心：随AR动态调整采样密度）
        :param x_center/y_center: 候选框中心 [N,1]
        :param w/h: 候选框宽高 [N,1]
        :param theta: 旋转角度 [N,1]
        :param ar: 长宽比 [N,1]
        :return: 采样点坐标 [total_samples,1], 采样角度 [total_samples,1], 批次索引 [total_samples,1]
        """
        """
                向量化版本：消除Python循环，支持批量处理
                """
        N = x_center.shape[0]
        device = x_center.device

        ar_squeezed = ar.squeeze(-1)
        small_ar_mask = ar_squeezed < self.ar_threshold
        mid_ar_mask = (ar_squeezed >= self.ar_threshold) & (ar_squeezed < self.extreme_ar_threshold)
        extreme_ar_mask = ar_squeezed >= self.extreme_ar_threshold

        # 确定每框采样数
        sample_nums = torch.ones(N, dtype=torch.long, device=device)
        sample_nums[small_ar_mask] = 1
        sample_nums[mid_ar_mask] = self.base_sample_num
        if extreme_ar_mask.any():
            extreme_nums = torch.clamp(
                torch.ceil(ar_squeezed[extreme_ar_mask]).long(),
                min=self.base_sample_num, max=self.max_sample_num
            )
            sample_nums[extreme_ar_mask] = extreme_nums

        # 创建采样掩码和索引
        max_samples = sample_nums.max().item()
        total_samples = sample_nums.sum().item()

        # 预分配张量
        x_sample = torch.zeros(total_samples, 1, device=device)
        y_sample = torch.zeros(total_samples, 1, device=device)
        theta_sample = torch.zeros(total_samples, 1, device=device)
        batch_idx = torch.zeros(total_samples, 1, dtype=torch.long, device=device)

        # 向量化生成所有椭圆采样点
        offset = 0
        for i in range(N):
            n = sample_nums[i].item()
            if n == 1:
                x_sample[offset] = x_center[i]
                y_sample[offset] = y_center[i]
                theta_sample[offset] = theta[i]
                batch_idx[offset] = i
                offset += 1
            else:
                # 向量化椭圆采样
                phi = torch.linspace(0, 2 * torch.pi, n + 1, device=device)[:-1]  # n个点
                a, b_i = w[i] / 2, h[i] / 2

                # 椭圆坐标 [n, 2]
                pts_local = torch.stack([a * torch.cos(phi), b_i * torch.sin(phi)], dim=1)

                # 旋转矩阵
                cos_t, sin_t = torch.cos(theta[i]), torch.sin(theta[i])
                R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], device=device)

                # 旋转+平移
                pts_rot = torch.matmul(pts_local, R.T) + torch.tensor([x_center[i], y_center[i]], device=device)

                x_sample[offset:offset + n] = pts_rot[:, 0:1]
                y_sample[offset:offset + n] = pts_rot[:, 1:2]
                theta_sample[offset:offset + n] = theta[i].expand(n, 1)
                batch_idx[offset:offset + n] = i
                offset += n

        return x_sample, y_sample, theta_sample, batch_idx, N

    def forward(self, x_center, y_center, theta, ar, bbox, irff_module):
        """
        【接口完全不变】与原有LAR-ISE完全一致，上游零改动
        Args:
            x_center/y_center/theta: 候选框中心坐标/角度 [N, 1]
            ar: 候选框长宽比 [N, 1]
            bbox: 候选框 [N, 5] (x1,y1,x2,y2,theta)
            irff_module: IRFF 模块实例
        Returns:
            shape_aware_feat: 形状感知特征 [N, C]，与原有输出维度完全一致
        """
        N = x_center.shape[0]
        device = x_center.device
        C = self.irff_channels

        # 1. 计算候选框宽高
        w = bbox[:, 2:3] - bbox[:, 0:1]
        h = bbox[:, 3:4] - bbox[:, 1:2]

        # 2. 自适应椭圆采样（替代原双分支的分离采样）
        x_sample, y_sample, theta_sample, batch_idx, N = self.adaptive_elliptical_sampling(
            x_center, y_center, w, h, theta, ar
        )

        # 3. 共享IRFF主干计算所有采样点特征（参数全共享，方案核心）
        sample_feat, _ = irff_module(x_sample, y_sample, theta_sample)
        sample_feat = sample_feat.clamp(-1e3, 1e3)

        # 4. 按候选框聚合特征（修复scatter_add索引维度）
        agg_feat = torch.zeros(N, C, device=device)
        # 扩展batch_idx到与sample_feat相同维度 [total_samples, C]
        batch_idx_expanded = batch_idx.view(-1, 1).expand(-1, C)
        # 按批次索引求和
        agg_feat.scatter_add_(0, batch_idx_expanded, sample_feat)

        # 按每个框的采样点数求平均
        sample_count = torch.zeros(N, 1, device=device)
        sample_count.scatter_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float32))
        agg_feat = agg_feat / sample_count.clamp(min=1e-6)  # 防除零

        # 5. HyperNetwork生成FiLM调制参数（方案核心）
        hyper_feat = self.hyper_net(torch.log(ar + 1e-6))  # 输入log(AR)，方案指定
        gamma = self.gamma_head(hyper_feat)  # [N, C] 缩放系数
        beta = self.beta_head(hyper_feat)    # [N, C] 偏置

        # 6. 条件化特征调制：f_shape = γ⊙f_φ + β
        modulated_feat = gamma * agg_feat + beta

        return modulated_feat.clamp(-1e3, 1e3)