# graph_cosine_similarity.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import matplotlib.pyplot as plt

class GraphCosineSim(nn.Module):
    """
    双输入单输出：图卷积余弦相似度模块
    输入:
        x: [B, C, H, W]   # 来自分支A（如 LSMA）
        y: [B, C, H, W]   # 来自分支B（如 CNN）
    输出:
        r: [B, 1, H, W]   # 相似度图 (0~1)

    结构:
    1) 1x1 投影 -> L2 归一化 -> 逐像素余弦相似度 s0∈[-1,1]，映射到 r0∈[0,1]
    2) 基于特征 (x', y', |x'-y'|, x'⊙y') 预测8方向门控权重 g∈[0,1]^{B,8,H,W}
    3) 图卷积式消息传递: 邻域聚合  r1 = (Σ_d g_d * shift_d(r0)) / (Σ_d g_d + eps)
       (可迭代 K 次；默认 K=1 或 2)
    4) 残差: r = (1-β)*r0 + β*r1
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 64,
        num_iters: int = 1,        # 图卷积迭代步数（1~2 通常足够）
        self_loop: bool = True,    # 是否包含自环
        beta: float = 0.7,         # 残差融合系数: r = (1-β)r0 + βr_gcn
        eps: float = 1e-6
    ):
        super().__init__()
        self.eps = eps
        self.num_iters = num_iters
        self.self_loop = self_loop
        self.beta = beta

        # 1x1 投影到嵌入维度
        self.proj_x = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.proj_y = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)

        # 生成门控（8方向 + 可选自环1个 = 9个）
        gate_in = embed_dim * 4  # [x', y', |x'-y'|, x'⊙y']
        self.num_dirs = 8 + (1 if self_loop else 0)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_in, gate_in // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(gate_in // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_in // 2, self.num_dirs, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()  # 输出 [0,1]
        )

        # 方向位移（8邻域：N,NE,E,SE,S,SW,W,NW），自环用 (0,0)
        self.shifts = [
            (-1,  0),  # N
            (-1,  1),  # NE
            ( 0,  1),  # E
            ( 1,  1),  # SE
            ( 1,  0),  # S
            ( 1, -1),  # SW
            ( 0, -1),  # W
            (-1, -1),  # NW
        ]
        if self_loop:
            self.shifts.append((0, 0))  # 自环

    @staticmethod
    def _l2_normalize(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # 按通道做 L2 归一化
        return z / (z.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)

    @staticmethod
    def _shift2d(t: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """
        使用 padding + 切片实现 2D 位移（零填充）
        t: [B, 1, H, W]
        """
        B, C, H, W = t.shape
        pad_t = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))  # (left,right,top,bottom)
        t = F.pad(t, pad_t, mode='constant', value=0.0)
        y1 = max(-dy, 0)
        y2 = y1 + H
        x1 = max(-dx, 0)
        x2 = x1 + W
        return t[:, :, y1:y2, x1:x2]

    def forward(self, x: torch.Tensor, y: torch.Tensor, sim_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x, y: [B,C,H,W]
        sim_prior: 可选先验相似度 [B,1,H,W]（比如来自上一层），会在残差里一并考虑
        return: r ∈ [0,1], [B,1,H,W]
        """
        B, C, H, W = x.shape
        assert y.shape == x.shape, "x and y must have same shape"

        # 1) 余弦相似度（像素级）
        x_emb = self._l2_normalize(self.proj_x(x))  # [B, E, H, W]
        y_emb = self._l2_normalize(self.proj_y(y))  # [B, E, H, W]
        s0 = (x_emb * y_emb).sum(dim=1, keepdim=True)          # [-1,1], [B,1,H,W]
        r0 = (s0 + 1.0) * 0.5                                  # [0,1]【b,1,h,w】

        # 2) 方向门控（边权）
        gate_in = torch.cat([x_emb, y_emb, (x_emb - y_emb).abs(), x_emb * y_emb], dim=1)
        gates = self.gate_conv(gate_in)                        # [B, num_dirs, H, W]
        # visualize_with_opencv(gates, output_path="./gates_heatmap.png", SIM_path="./gates_sim.png")
        # 如果有 sim_prior，可作为额外的自环增强（可选）
        if sim_prior is not None:
            sp = torch.sigmoid(sim_prior)                      # 保证在 [0,1]
            if self.self_loop:
                gates[:, -1:] = torch.clamp(gates[:, -1:] + 0.25 * sp, 0.0, 1.0)

        # 3) 图卷积消息传递（K 次）
        r = r0  # 初始值
        for _ in range(self.num_iters):     #每一次迭代，都会让相似度在邻居间传递。
            # 邻居聚合
            agg = 0.0
            deg = 0.0
            for d, (dy, dx) in enumerate(self.shifts):
                g = gates[:, d:d+1]                     #               # [B,1,H,W]  每个像素对该方向邻居的“权重”# 当前方向的门控权重 [B,1,H,W]
                nbr = self._shift2d(r, dy, dx)                 # [B,1,H,W]  # 把 r 整体平移 (dy,dx)，获得当前方向的邻居值。
                agg = agg + g * nbr     #累积邻居贡献（权重 × 值）。
                deg = deg + g       #累积邻居权重之和。
            r_gcn = agg / (deg + self.eps)                     # 归一化聚合
            r = (1.0 - self.beta) * r + self.beta * r_gcn      # 残差融合

        # 保证范围 [0,1]
        r = torch.clamp(r, 0.0, 1.0)
        return r
if __name__ == "__main__":
    import torch
    B, C, H, W = 3, 256, 128, 128
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)

    gcs = GraphCosineSim(in_channels=C, embed_dim=64, num_iters=2, self_loop=True, beta=0.7)
    r = gcs(x, y)  # [B,1,H,W]，范围 [0,1]
    print("r.shape:", r.shape, r.min().item(), r.max().item())
