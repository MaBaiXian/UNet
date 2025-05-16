import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """两个连续的Conv + BN + ReLU"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # 第一个卷积层，使用 3x3 卷积核，padding 为 1 以保持特征图尺寸不变
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 批量归一化层
            nn.BatchNorm2d(out_channels),
            # ReLU 激活函数
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样模块: MaxPool + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            # 最大池化层，核大小为 2，步长为 2，将特征图尺寸缩小一半
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """上采样模块: TransposeConv + 拼接 + DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # 转置卷积层，用于上采样，将特征图尺寸扩大一倍
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # DoubleConv 模块，处理拼接后的特征图
        self.conv = DoubleConv(in_channels, out_channels)  # in_channels 为拼接后通道

    def forward(self, x1, x2):
        # 第一步：上采样操作
        x1 = self.up(x1)
        # 第二步：计算特征图尺寸差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # 第三步：对 x1 进行填充操作
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 第四步：拼接特征图
        x = torch.cat([x2, x1], dim=1)
        # 第五步：通过 DoubleConv 模块处理拼接后的特征图
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # 1x1 卷积层，用于调整通道数
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # 输入卷积层，将输入图像转换为 64 通道的特征图
        self.in_conv = DoubleConv(n_channels, 64)
        # 四个下采样模块，通道数依次翻倍
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 四个上采样模块，通道数依次减半
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # 输出卷积层，将特征图的通道数转换为类别数
        self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        # 编码器部分
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器部分
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits
