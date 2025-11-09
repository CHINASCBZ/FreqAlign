import torch
from torch import nn
# from .common import LayerNorm2d
from torch.nn import functional as F
#-----------------------------------Multi scale frequency domain interaction adapter多尺度频域交互adapter--------------------------------------------------#
# ---------------- SE Block ----------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ---------------- 空洞卷积 + SE Block ----------------
class FreqEnhanceBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            SELayer(channels)
        )

    def forward(self, x):
        return self.block(x)

# ---------------- 多尺度 PPM ----------------
class ModifyPPM_MSFDI(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(reduction_dim, reduction_dim, kernel_size=1, bias=False, groups=reduction_dim),
                nn.GELU()
            ) for bin in bins
        ])
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, bias=False, groups=in_dim),
            nn.Conv2d(in_dim, in_dim // 4, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):
        out_list = [self.local_conv(x)]
        for f in self.features:
            out_list.append(f(x))
        return out_list  # 共4个尺度

# ---------------- Cross Attention ----------------
class CrossScaleFreqAttention(nn.Module):
    def __init__(self, token_dim, freq_dim, out_dim):
        super().__init__()
        self.q_proj = nn.Conv2d(token_dim, out_dim, 1)
        self.kv_proj = nn.Conv2d(freq_dim, out_dim * 2, 1)
        self.attn_proj = nn.Conv2d(out_dim, out_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, token_feat, freq_feat):
        B, _, H, W = token_feat.shape
        q = self.q_proj(token_feat).flatten(2).transpose(1, 2)
        kv = self.kv_proj(freq_feat)
        k, v = kv.chunk(2, dim=1)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        attn = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        attn = self.softmax(attn)
        out = torch.bmm(attn, v).transpose(1, 2).view(B, -1, H, W)
        return self.attn_proj(out)

# ---------------- 主模块 ----------------
class MSFDIadapter(nn.Module):
    def __init__(self, in_dim, hidden_dim, patch_num):
        super().__init__()
        self.down_project = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.up_project = nn.Linear(hidden_dim, in_dim)
        self.patch_num = patch_num

        self.conv = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.mppm = ModifyPPM_MSFDI(256, hidden_dim // 4, bins=[32, 64, 128])

        # 四个尺度的增强分支
        self.freq_enhancers = nn.ModuleList([
            FreqEnhanceBlock(hidden_dim // 4, d) for d in [1, 3, 5, 7]
        ])

        # 四个尺度的 attention 分支
        self.scale_attentions = nn.ModuleList([
            CrossScaleFreqAttention(hidden_dim, hidden_dim // 4, hidden_dim)
            for _ in range(4)
        ])

        # 融合：concat(attn, enhanced) → conv1x1
        fusion_in_channels = hidden_dim + (hidden_dim // 4)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, hidden_dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, x, LF):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        token_feat = self.act(self.down_project(x)).permute(0, 3, 1, 2).contiguous()  # [B, hidden_dim, H, W]

        # 提取多尺度频域特征
        LF_feat = self.conv(LF)
        freq_feats = self.mppm(LF_feat)  # [4 x B, C', h_i, w_i]

        attn_feats = []
        enhanced_feats = []

        for i in range(4):
            freq_feat = freq_feats[i]

            # Attn 分支
            attn_out = self.scale_attentions[i](token_feat, freq_feat)
            attn_feats.append(attn_out)

            # 增强分支
            enhanced = self.freq_enhancers[i](freq_feat)
            enhanced = F.interpolate(enhanced, size=token_feat.shape[2:], mode='bilinear', align_corners=True)
            enhanced_feats.append(enhanced)

        # 分支聚合
        attn_fused = torch.mean(torch.stack(attn_feats), dim=0)        # [B, hidden_dim, H, W]
        freq_fused = torch.mean(torch.stack(enhanced_feats), dim=0)    # [B, hidden_dim//4, H, W]

        fusion = torch.cat([attn_fused, freq_fused], dim=1)  # [B, hidden_dim + hidden_dim//4, H, W]
        fusion = self.fusion_conv(fusion)

        # 恢复维度 + 残差连接
        out = fusion.permute(0, 2, 3, 1).contiguous()
        out = self.up_project(out)
        return out + x



if __name__ == '__main__':
    model=MSFDIadapter(in_dim=256, hidden_dim=256, patch_num=32)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型的总参数量为: {total_params}")