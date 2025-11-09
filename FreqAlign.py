import torch
import torch.nn as nn
import torch.nn.functional as F
from geoseg.models.MDSAM_AAAB import MDSAM
from typing import Tuple
from geoseg.models.GCCSM import GraphCosineSim
from geoseg.models.DCAB import DCAB
# import pywt
import math
#---------------------------------unet内部代码--------------------------------#
class ConvBNReLUx2(nn.Module):
    '''
    卷积模块,包含CONV-BN-RELU-CONV-BN-RELU.
    '''

    def __init__(self, in_channels, out_channels):
        super(ConvBNReLUx2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class UpSample(nn.Module):
    '''
    上采样模块,包含UP2x-CONV-BN-RELU
    '''

    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

#---------------------------------unet内部代码--------------------------------#

#---------------------------------小波变换分解代码--------------------------------#
class HaarDWT2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        h0 = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)])
        h1 = torch.tensor([1/math.sqrt(2),-1/math.sqrt(2)])
        ll = torch.einsum('i,j->ij', h0, h0)  # 2x2
        lh = torch.einsum('i,j->ij', h0, h1)
        hl = torch.einsum('i,j->ij', h1, h0)
        hh = torch.einsum('i,j->ij', h1, h1)
        # 4C x 1 x 2 x 2, groups=C
        W = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)         # 4 x 1 x 2 x 2
        W = W.repeat(channels, 1, 1, 1)                                # 4C x 1 x 2 x 2
        self.register_buffer('weight', W)
        self.channels = channels

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # 如果H/W为奇数，镜像pad到偶数，避免频谱偏差
        pad_h = (H % 2)
        pad_w = (W % 2)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        y = F.conv2d(x, self.weight, stride=2, groups=self.channels)   # B x 4C x H/2 x W/2
        LL  = y[:, 0:C, ...]
        LH  = y[:, 1*C:2*C, ...]
        HL  = y[:, 2*C:3*C, ...]
        HH  = y[:, 3*C:4*C, ...]
        HF = torch.cat([LH, HL, HH], dim=1)
        return LL, HF



#---------------------------------小波变换分解代码--------------------------------#

class AAABnet(nn.Module):
    def __init__(self, img_size=512,in_channels=3, num_classes=1):  # 降低16倍
        super().__init__()
        self.dwt = HaarDWT2D(in_channels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        # weights_path = '../ckpts/sam_vit_b_01ec64.pth'  # 替换为你保存权重文件的路径
        # state_dict = torch.load(weights_path, map_location=self.device)

        # 实例化 MDSAM 模型并加载权重
        self.MDSAM = MDSAM(img_size=img_size) # 确保模型实例化
        # self.MDSAM.load_state_dict(state_dict, strict=False)  # 使用 strict=False 忽略不匹配的层
        #---------------------------------unet参数设置-------------------------------#
        base_channel = 64
        channels = [base_channel, base_channel * 2, base_channel * 4, base_channel * 8, base_channel * 16]

        self.Maxpool2 = nn.MaxPool2d(2, 2)
        self.Maxpool3 = nn.MaxPool2d(2, 2)
        self.Maxpool4 = nn.MaxPool2d(2, 2)
        self.Maxpool5 = nn.MaxPool2d(2, 2)

        self.Conv1 = ConvBNReLUx2(in_channels+9, channels[0])  # stem
        self.Conv2 = ConvBNReLUx2(channels[0], channels[1])  #
        self.Conv3 = ConvBNReLUx2(channels[1], channels[2])  #
        self.Conv4 = ConvBNReLUx2(channels[2], channels[3])  #
        self.Conv5 = ConvBNReLUx2(channels[3], channels[4])  #

        self.Up5 = UpSample(channels[4], channels[3])
        self.Up5_Conv = ConvBNReLUx2(channels[4], channels[3])

        self.Up4 = UpSample(channels[3], channels[2])
        self.Up4_Conv = ConvBNReLUx2(channels[3], channels[2])

        self.Up3 = UpSample(channels[2], channels[1])
        self.Up3_Conv = ConvBNReLUx2(channels[2], channels[1])

        self.Up2 = UpSample(channels[1], channels[0])
        self.Up2_Conv = ConvBNReLUx2(channels[1], channels[0])

        self.out = nn.Conv2d(channels[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.conv_last = nn.Sequential(
            nn.Conv2d(channels[0], num_classes, kernel_size=1, stride=1, padding=0),
        )
        # ---------------------------------unet参数设置-------------------------------#

        # ---------------------------------SAM层级特征通道尺寸设置------------------------------#
        self.a_conv = ConvBNReLUx2(256, 64)
        self.b_conv = ConvBNReLUx2(256, 128)
        self.c_conv = ConvBNReLUx2(256, 256)
        self.d_conv = ConvBNReLUx2(256, 512)
        # ---------------------------------SAM层级特征通道尺寸设置-------------------------------#




        # ---------------------------------GCCSM参数设置-------------------------------#
        self.GCCSM1 = GraphCosineSim(in_channels=64, embed_dim=64, num_iters=2, self_loop=True, beta=0.7)
        self.GCCSM2 = GraphCosineSim(in_channels=128, embed_dim=64, num_iters=2, self_loop=True, beta=0.7)
        self.GCCSM3 = GraphCosineSim(in_channels=256, embed_dim=64, num_iters=2, self_loop=True, beta=0.7)
        self.GCCSM4 = GraphCosineSim(in_channels=512, embed_dim=64, num_iters=2, self_loop=True, beta=0.7)


        #-------------------------------------DCAB参数设置-------------------------------#
        mulit_channels = [128, 256, 512, 512]
        # sizes = [(256, 256), (128, 128), (64, 64), (32, 32)]
        self.DCAB = DCAB(dim=512, in_channels=mulit_channels, num_points=4,drop=0.1)

        #-------------------------------------小波变换设置-------------------------------#




    def forward(self, img):  # 输入尺寸 (1, 3, 512, 512)
        #不同尺度频域特征
        LF, HF = self.dwt(img)
        HF = F.interpolate(HF, size=(512, 512), mode="bilinear", align_corners=False)
        '''
        LF:1,3,256,256
        HF:1,9,256,256
        '''



        # 获取 MDSAM 的特征图列表
        # LF =
        features_list = self.MDSAM(img,LF)
        a, b, c, d = features_list
        #----------------SAM尺度配准-------------------#
        a = self.a_conv(a)
        b = self.b_conv(b)
        c = self.c_conv(c)
        d = self.d_conv(d)
        a = F.interpolate(a, size=(256, 256), mode="bilinear", align_corners=False)     #1,64,256,256
        b = F.interpolate(b, size=(128, 128), mode="bilinear", align_corners=False) #1,128,128,128
        c = F.interpolate(c, size=(64, 64), mode="bilinear", align_corners=False)#1,256,64,64
        d = F.interpolate(d, size=(32, 32), mode="bilinear", align_corners=False)#1,512,32,32
        # ----------------SAM尺度配准-------------------#


        #-----------unet编码器--------------------#
        # stem

        e1 = self.Conv1(torch.cat([img, HF],dim=1)) # 256*2/3->256*2/64

        #
        e2 = self.Maxpool2(e1)  #
        e22 = self.GCCSM1(a,e2)     #1,1,256,256
        e2 = self.Conv2(e2)  # 256/64->128/128

        #
        e3 = self.Maxpool3(e2)  #
        e33 = self.GCCSM2(b,e3)
        e3 = self.Conv3(e3)  # 128/128 -> 64/256
        #
        e4 = self.Maxpool4(e3)
        e44 = self.GCCSM3(c,e4)
        e4 = self.Conv4(e4)  # 64/256->32/512
        #
        e5 = self.Maxpool5(e4)
        e55 = self.GCCSM4(d,e5)
        e5 = self.DCAB(d, [e2, e3, e4,e5], sim=e55, tau0=1.0)
        e5 = self.Conv5(e5)  # 32/512->16/1024

        #
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up5_Conv(d5)*e44
        #
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up4_Conv(d4)*e33
        #
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up3_Conv(d3)*e22
        #
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up2_Conv(d2)

        # out = self.out(d2)
        out = self.conv_last(d2)

        return out

if __name__ == '__main__':
    with torch.no_grad():  # 不进行梯度计算，减少内存消耗
        # 检查设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"当前设备: {device}")

        # 初始化模型并移动到设备
        model = AAABnet(512,3,1).to(device)

        # 打印模型总参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型的总参数量为: {total_params}")

        # 创建测试输入
        x = torch.randn(1, 3,512, 512).to(device)  # 模拟一个批量的RGB图像

        # 前向传播测试
        output = model(x)

        # 打印输出形状
        print("输出的形状:", output.shape)  # 输出是最后一层特征图的形状
