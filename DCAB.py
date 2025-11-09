# dcab.py
# -*- coding: utf-8 -*-
"""
Deformable Cross-Attention Bridge (DCAB)
- 可变形跨尺度采样
- 相似度温度门控（tau = tau0 / (r + eps)）
- 残差 + FFN
- 兼容任意多尺度输入通道（in_channels），内部 1x1 对齐到 dim

输入:
  S          : [B, C, Hq, Wq]          # LSMA 顶层语义特征（Query来源）
  cnn_feats  : list[Tensor]             # 每层 [B, C_l, H_l, W_l]（Key/Value来源）
  sim (可选) : [B,1,Hq,Wq] 或 [B,Hq,Wq] 或 [B,Nq] 或 [B,Nq,1] 的相似度得分 r∈[0,1]
输出:
  Z          : [B, C, Hq, Wq]          # 融合后的颈部特征，送入解码器
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


#-----------------------------------------可视化代码，训练时注销-----------------------------------------------------------#

#方案一：伪彩色变换+下载,直接将tensor中的第一层打印出来。
import cv2
import numpy as np




class MLP(nn.Module):
    def __init__(self, dim: int, hidden: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        hidden = hidden or 4 * dim
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DCAB(nn.Module):
    """
    DCAB with similarity-temperature & per-level channel adapters.

    Args:
        dim (int): 统一维度 C（Q/V 的通道）
        in_channels (List[int]): 每个尺度原始通道数（会用 1x1 conv 对齐到 C）
        num_heads (int): 注意力头数 H
        num_points (int): 每尺度采样点数 M
        drop (float): dropout 概率
        align_corners (bool): grid_sample 的 align_corners

    Forward:
        S: [B, C, Hq, Wq]
        cnn_feats: List[[B, C_l, H_l, W_l]]，长度 = L = len(in_channels)
        sim: Optional[Tensor]  (形状见文件头注释)
        tau0: float
        return Z: [B, C, Hq, Wq]
    """

    def __init__(
        self,
        dim: int,
        in_channels: List[int],
        num_heads: int = 4,
        num_points: int = 4,
        drop: float = 0.0,
        align_corners: bool = True,
    ):
        super().__init__()
        self.C = dim
        self.L = len(in_channels)
        self.Hh = num_heads
        self.M = num_points
        self.align_corners = align_corners

        # 每尺度 1x1 对齐到 C（同时作为 V 投影）
        self.v_proj = nn.ModuleList([nn.Conv2d(cin, dim, 1) for cin in in_channels])

        # Q 投影
        self.q_proj = nn.Linear(dim, dim)

        # 偏移 / 权重 预测 (一次性预测 H*L*M 组)
        self.offset_pred = nn.Linear(dim, self.Hh * self.L * self.M * 2)  # 2: (dx,dy)
        self.weight_pred = nn.Linear(dim, self.Hh * self.L * self.M)

        # 输出投影 & FFN
        self.out_proj = nn.Linear(self.Hh * dim, dim)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_out = nn.LayerNorm(dim)
        self.ffn = MLP(dim, drop=drop)
        self.drop = nn.Dropout(drop)

        self._reset()

    def _reset(self):
        # 让初始行为接近 “参考点 + 均匀权重”
        nn.init.zeros_(self.offset_pred.weight)
        nn.init.zeros_(self.offset_pred.bias)
        nn.init.zeros_(self.weight_pred.bias)

    @staticmethod
    def _ref_points(Hq: int, Wq: int, device) -> torch.Tensor:
        """返回 [1, Nq, 2] 的中心网格 (x,y) ∈ [0,1]"""
        yy, xx = torch.meshgrid(
            torch.linspace(0.5 / Hq, 1 - 0.5 / Hq, Hq, device=device),
            torch.linspace(0.5 / Wq, 1 - 0.5 / Wq, Wq, device=device),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1).view(1, Hq * Wq, 2)  # [1,Nq,2]

    @staticmethod
    # def _to_gridm11(grid01: torch.Tensor) -> torch.Tensor:        #原始代码
    #     """[0,1] -> [-1,1]（grid_sample 需要）"""
    #     g = grid01.clone()
    #     g[..., 0] = g[..., 0] * 2.0 - 1.0
    #     g[..., 1] = g[..., 1] * 2.0 - 1.0
    #     return g

    def _to_gridm11(grid01: torch.Tensor) -> torch.Tensor:      #进一步修改的代码
        """[0,1] -> [-1,1]，并确保网格值在 [-1, 1] 范围内"""
        # 直接使用 tanh 对 grid01 进行限制
        g = grid01.clone()
        g[..., 0] = torch.tanh(g[..., 0] * 2.0 - 1.0)  # x 坐标从 [0,1] -> [-1,1]，并做 tanh 限制
        g[..., 1] = torch.tanh(g[..., 1] * 2.0 - 1.0)  # y 坐标从 [0,1] -> [-1,1]，并做 tanh 限制

        # 确保网格坐标位于 [-1, 1] 范围内
        g = g.clamp(min=-1.0, max=1.0)

        return g


    def forward(
        self,
        S: torch.Tensor,
        cnn_feats: List[torch.Tensor],
        sim: Optional[torch.Tensor] = None,
        tau0: float = 1.0,
        return_extra: bool = False,
    ) -> torch.Tensor:
        B, C, Hq, Wq = S.shape
        assert len(cnn_feats) == self.L, f"expect {self.L} levels, got {len(cnn_feats)}"

        # ---- 1) Q 投影 & 参考点 ----
        Nq = Hq * Wq
        S_seq = S.flatten(2).transpose(1, 2)      # [B, Nq, C]
        Q    = self.ln_q(self.q_proj(S_seq))         # [B, Nq, C]
        ref = self._ref_points(Hq, Wq, S.device)  # [1, Nq, 2]  #生成 Query 特征图上每个位置的“参考点坐标”，
        # ---- 2) V（通道对齐） ----
        V = [self.v_proj[l](cnn_feats[l]) for l in range(self.L)]  # each [B,C,Hl,Wl]

        # ---- 3) 偏移/权重预测 + 相似度温度门控 ----
        offsets = self.offset_pred(Q).view(B, Nq, self.Hh, self.L, self.M, 2)   # [B,Nq,Hh,L,M,2]       #m表示每个查询点附近采样的采样点数量
        weights = self.weight_pred(Q).view(B, Nq, self.Hh, self.L, self.M)      # [B,Nq,Hh,L,M]
        #------可视化代码---------#

        # ------可视化代码---------#


        if sim is not None:
            # 将 sim 变为 [B, Nq, 1]，再扩到 [B,Nq,1,1,1] 以匹配 weights
            if sim.dim() == 4:  # [B,1,Hq,Wq] or [B,Hq,Wq]
                sim = sim if sim.shape[1] != 1 else sim[:, 0]  # [B,Hq,Wq]
                sim = sim.reshape(B, Nq, 1)                     # [B,Nq,1]
            elif sim.dim() == 2:  # [B,Nq]
                sim = sim.unsqueeze(-1)                         # [B,Nq,1]
            # tau & 广播
            tau = tau0 / (sim + 1e-3)                           # [B,Nq,1]
            tau = tau.view(B, Nq, 1, 1, 1)                      # ✅ 与 [B,Nq,Hh,L,M] 广播
            weights = weights / tau

        # (L*M) 维度上 softmax
        weights = weights.flatten(-2)                            # [B,Nq,Hh,L*M]
        weights = torch.softmax(weights, dim=-1)
        weights = weights.view(B, Nq, self.Hh, self.L, self.M)   # [B,Nq,Hh,L,M]

        # ---- 4) 可变形跨尺度采样 ----
        outs = []
        for h in range(self.Hh):
            acc = 0.0
            for l in range(self.L):
                # 参考点 + 偏移，注意把 ref 扩成 [B,Nq,1,2] 与 offsets 对齐
                grid01 = ref.expand(B, -1, -1).unsqueeze(2) + offsets[:, :, h, l]  # [B,Nq,M,2]  (1,1024,4,2):1024表示查询点，4表示每个查询点的采样数，2表示X,Y轴

                # 将 grid01 转换为 [-1,1] 的坐标网格
                grid = self._to_gridm11(grid01).view(B, Nq * self.M, 1, 2)         # [B,Nq*M,1,2]   #这行代码的作用是将一个归一化的坐标网格（范围 [0, 1]）转换为适用于 F.grid_sample 函数的格式（范围 [-1, 1]）

                # 双线性采样
                sampled = F.grid_sample(
                    V[l], grid, mode="bilinear", padding_mode="zeros",
                    align_corners=self.align_corners
                )  # [B,C,Nq*M,1] (1,512,4096,1)
                sampled = sampled.view(B, self.C, Nq, self.M).permute(0, 2, 3, 1)  # [B,Nq,M,C]

                # 权重融合
                w = weights[:, :, h, l].unsqueeze(-1)                               # [B,Nq,M,1]
                acc = acc + (sampled * w).sum(dim=2)                              # [B,Nq,C]
            outs.append(acc)                                                        # list of [B,Nq,C]
        Y = torch.cat(outs, dim=-1)              # [B,Nq,Hh*C]
        Y = self.drop(self.out_proj(Y))          # [B,Nq,C]



        # ---- 5) 残差 + FFN ----
        Z = self.ln_out(S_seq + Y)               # [B,Nq,C]
        Z = Z + self.ffn(Z)                      # [B,Nq,C]
        Z = Z.transpose(1, 2).view(B, C, Hq, Wq)


        return Z


# ======================= 自检示例 =======================
if __name__ == "__main__":
    torch.manual_seed(0)

    # 假设 4 个尺度
    B, C = 1, 64        #最终输出通道数
    Hq, Wq = 32, 32     #最终输出分辨率
    in_channels = [128, 256, 256, 512]
    sizes = [(128, 128), (64, 64), (32, 32), (16, 16)]

    # 构造假数据
    S = torch.randn(B, 512, Hq, Wq)
    cnn_feats = [torch.randn(B, c, h, w) for (c, (h, w)) in zip(in_channels, sizes)]
    sim = torch.rand(B, 1, Hq, Wq)  # 或者 None

    # 实例化并前向
    dcab = DCAB(dim=512, in_channels=in_channels, num_heads=4, num_points=4, drop=0.1)
    Z = dcab(S, cnn_feats, sim=sim, tau0=1.0)

    print("Input  S:", S.shape)         # [B, C, Hq, Wq]
    print("Output Z:", Z.shape)         # [B, C, Hq, Wq]
