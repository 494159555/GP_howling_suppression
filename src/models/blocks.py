"""U-Net 公共构建块

编码器/解码器层的工厂函数，消除各模型间的重复定义。
"""

import torch.nn as nn


def make_encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """编码器下采样块

    Conv2d(stride=(2,1)) + BatchNorm2d + LeakyReLU(0.2)

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数

    Returns:
        nn.Sequential 编码器块
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def make_decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """解码器上采样块

    ConvTranspose2d(stride=(2,1)) + BatchNorm2d + ReLU

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数

    Returns:
        nn.Sequential 解码器块
    """
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0),
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def make_output_block(in_channels: int) -> nn.Sequential:
    """输出块

    ConvTranspose2d(stride=(2,1)) + Sigmoid，生成 [0,1] 掩膜。

    Args:
        in_channels: 输入通道数

    Returns:
        nn.Sequential 输出块
    """
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, 1,
            kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0),
        ),
        nn.Sigmoid(),
    )
