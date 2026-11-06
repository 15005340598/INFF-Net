import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.builder import ROTATED_BACKBONES


class MultiResHashEncoding(nn.Module):
    """
    多分辨率哈希编码模块，严格对齐方案参数
    L=16层哈希网格，每层T=2^16个8维可学习特征向量
    分辨率从 2^4 到 2^19 指数增长
    """

    def __init__(self,
                 num_levels=16,  # L=16 层哈希网格
                 feat_dim=8,  # 每个哈希条目8维特征
                 table_size=2 ** 16,  # 每层哈希表大小 T=2^16
                 min_res=2 ** 4,  # 最小分辨率 2^4
                 max_res=2 ** 19,  # 最大分辨率 2^19
                 img_scale=1024):  # 输入图像尺寸（DOTA固定resize到1024）
        super().__init__()
        self.num_levels = num_levels
        self.feat_dim = feat_dim
        self.table_size = table_size
        self.img_scale = img_scale

        # 计算每层的分辨率因子（指数增长）
        self.b = torch.exp((torch.log(torch.tensor(max_res)) - torch.log(torch.tensor(min_res))) / (num_levels - 1))
        self.resolutions = min_res * (self.b ** torch.arange(num_levels))

        # 哈希质数（防碰撞，行业通用值）
        self.primes = torch.tensor([1, 2654435761, 805459861], dtype=torch.long)

        # 可学习哈希表：[num_levels, table_size, feat_dim]
        self.hash_table = nn.Parameter(
            torch.randn(num_levels, table_size, feat_dim) * 0.01  # 小初始化
        )

    def forward(self, x, y):
        """
        空间坐标哈希编码
        Args:
            x/y: 像素坐标，shape [N, 1]
        Returns:
            hash_feat: 拼接后的哈希特征，shape [N, num_levels*feat_dim=128]
        """
        device = x.device
        N = x.shape[0]
        self.primes = self.primes.to(device)
        self.resolutions = self.resolutions.to(device)

        # 1. 像素坐标归一化到 [0, 1] 区间（适配哈希网格）
        x_norm = x / self.img_scale  # [N,1]
        y_norm = y / self.img_scale  # [N,1]
        coords = torch.cat([x_norm, y_norm], dim=-1)  # [N,2]

        # 2. 逐层计算哈希编码
        level_feats = []
        for level in range(self.num_levels):
            # 缩放坐标到当前层级分辨率
            scaled = coords * self.resolutions[level]
            # 取网格整数索引
            grid_floor = torch.floor(scaled).long()
            grid_ceil = torch.ceil(scaled).long()

            # 三线性插值的8个顶点（2D用4个顶点即可）
            corners = torch.stack([
                grid_floor,
                torch.stack([grid_floor[:, 0], grid_ceil[:, 1]], dim=-1),
                torch.stack([grid_ceil[:, 0], grid_floor[:, 1]], dim=-1),
                grid_ceil
            ], dim=1)  # [N,4,2]

            # 计算每个顶点的哈希索引
            hash_idx = (
                               (corners[..., 0] * self.primes[0]) ^
                               (corners[..., 1] * self.primes[1]) ^
                               (level * self.primes[2])
                       ) % self.table_size  # [N,4]

            # 查表得到顶点特征
            corner_feats = self.hash_table[level, hash_idx, :]  # [N,4,8]

            # 双线性插值权重
            frac = scaled - grid_floor  # [N,2]
            weight_x = frac[:, 0:1]  # [N,1]
            weight_y = frac[:, 1:2]  # [N,1]

            # 双线性插值计算当前坐标的特征
            w00 = (1 - weight_x) * (1 - weight_y)  # [N,1]
            w01 = (1 - weight_x) * weight_y  # [N,1]
            w10 = weight_x * (1 - weight_y)  # [N,1]
            w11 = weight_x * weight_y  # [N,1]
            weights = torch.stack([w00, w01, w10, w11], dim=1)  # [N,4,1]

            # 加权求和得到当前层级的特征
            level_feat = (corner_feats * weights).sum(dim=1)  # [N,8]
            level_feats.append(level_feat)

        # 3. 拼接所有层级的特征，输出 [N, 16*8=128]
        hash_feat = torch.cat(level_feats, dim=-1)
        return hash_feat

@ROTATED_BACKBONES.register_module()  # 关键：注册到 ROTATED_BACKBONES
class ImplicitRotationFeatureField(nn.Module):
    """
    隐式旋转特征场 (IRFF) 模块
    核心功能：生成旋转等变的隐式特征，支持旋转等变约束损失
    """

    def __init__(self,
                 in_channels=256,  # 输出特征维度，保持原有256不变，兼容下游
                 # 哈希编码参数
                 hash_num_levels=16,
                 hash_feat_dim=8,
                 hash_table_size=2 ** 16,
                 min_res=2 ** 4,
                 max_res=2 ** 19,
                 img_scale=1024,
                 # 角度编码参数
                 angle_freq_num=4,  # 方案中k=0~3，共4个频率
                 # 等变约束开关
                 with_rotation_equivariant=True):
        super().__init__()
        self.in_channels = in_channels
        self.with_rotation_equivariant = with_rotation_equivariant

        # ====================== 方案核心替换 ======================
        # 1. 多分辨率哈希编码（替换原有的位置编码+大MLP）
        self.hash_encoder = MultiResHashEncoding(
            num_levels=hash_num_levels,
            feat_dim=hash_feat_dim,
            table_size=hash_table_size,
            min_res=min_res,
            max_res=max_res,
            img_scale=img_scale
        )

        # 2. 角度紧凑傅里叶编码（方案指定的γ(θ)）
        self.angle_freq_num = angle_freq_num
        self.angle_encoding_dim = 2 * angle_freq_num  # 4个频率→8维

        # 3. 微型特征精炼网络（Tiny MLP，方案指定2层，隐藏层64维）
        mlp_input_dim = hash_num_levels * hash_feat_dim + self.angle_encoding_dim  # 128+8=136
        self.tiny_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, in_channels)  # 输出保持256维，完全兼容原有逻辑
        )
        # ==========================================================

    def angle_encoding(self, theta):
        """方案指定的紧凑傅里叶角度编码：γ(θ) = [sin(2^kθ), cos(2^kθ)]_{k=0}^3"""
        encoding = []
        for k in range(self.angle_freq_num):
            scale = 2 ** k
            encoding.append(torch.sin(scale * theta))
            encoding.append(torch.cos(scale * theta))
        return torch.cat(encoding, dim=-1)  # [N, 8]

    def forward(self, x, y, theta, compute_equiv=False):
        """
        【接口完全不变】和原有IRFF完全一致，下游零改动
        Args:
            x/y: 空间坐标，shape (N, 1)
            theta: 旋转角度，shape (N, 1)
        Returns:
            feat: 输出特征 (N, 256)，和原有维度完全一致
            equiv_loss: 等变约束损失 (标量)
        """
        # 1. 哈希编码 + 角度编码
        hash_feat = self.hash_encoder(x, y)  # [N, 128]
        angle_enc = self.angle_encoding(theta)  # [N, 8]
        input_feat = torch.cat([hash_feat, angle_enc], dim=-1)  # [N, 136]

        # 2. 微型MLP精炼特征
        feat = self.tiny_mlp(input_feat)  # [N, 256]

        # 3. 旋转等变约束损失（完全保留原有逻辑，适配方案的等变要求）
        equiv_loss = torch.tensor(0.0, device=feat.device)
        if compute_equiv and self.training and self.with_rotation_equivariant:
            # 随机生成旋转角度 α
            alpha = torch.rand_like(theta) * torch.pi
            # 构建旋转矩阵 R(α)
            R_alpha = torch.stack([
                torch.cat([torch.cos(alpha), -torch.sin(alpha)], dim=-1),
                torch.cat([torch.sin(alpha), torch.cos(alpha)], dim=-1)
            ], dim=1)  # (N, 2, 2)

            # 旋转坐标: p' = R(α) * p
            p = torch.stack([x, y], dim=-1)  # (N, 2)
            p_squeezed = p.squeeze(1)  # 兼容多余维度
            p_3d = p_squeezed.unsqueeze(-1)
            p_rot = torch.bmm(R_alpha, p_3d).squeeze(-1)  # (N, 2)
            x_rot, y_rot = p_rot[:, 0:1], p_rot[:, 1:2]
            theta_rot = theta + alpha  # 角度同步旋转

            # 计算旋转后的特征
            hash_feat_rot = self.hash_encoder(x_rot, y_rot)
            angle_enc_rot = self.angle_encoding(theta_rot)
            feat_rot = self.tiny_mlp(torch.cat([hash_feat_rot, angle_enc_rot], dim=-1))

            # 特征旋转: feat' = R(α) * feat（对齐方案的等变约束）
            B, C = feat.shape
            feat_2d = feat.reshape(B, 2, C // 2)
            feat_rot_2d = torch.bmm(R_alpha, feat_2d)
            feat_rotated = feat_rot_2d.reshape(B, C)

            # 等变约束损失（MSE）
            equiv_loss = F.mse_loss(feat_rot, feat_rotated)
            #print(f"[IRFF] 单样本 equiv_loss: {equiv_loss.item():.6f}")

        return feat, equiv_loss