"""U-Net v1 - 3层轻量级U-Net

编码器3层下采样 + 解码器3层上采样 + 跳跃连接。
参数量约1.2M，适合快速实验。
"""

import torch
import torch.nn as nn

from .blocks import make_encoder_block, make_decoder_block, make_output_block


class AudioUNet3(nn.Module):
    """3层U-Net音频啸叫抑制模型

    输入: [batch, 1, 256, time]
    输出: [batch, 1, 256, time]
    """

    def __init__(self):
        super(AudioUNet3, self).__init__()

        # 编码器: 1→16→32→64
        self.enc1 = make_encoder_block(1, 16)
        self.enc2 = make_encoder_block(16, 32)
        self.enc3 = make_encoder_block(32, 64)

        # 解码器: 64→32→16→1 (输入通道含跳跃连接拼接)
        self.dec3 = make_decoder_block(64, 32)
        self.dec2 = make_decoder_block(64, 16)   # 输入64 = dec3的32 + enc2的32
        self.dec1 = make_output_block(32)          # 输入32 = dec2的16 + enc1的16

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)

        # 编码器
        e1 = self.enc1(x_log)    # [B, 16, 128, T]
        e2 = self.enc2(e1)       # [B, 32, 64, T]
        e3 = self.enc3(e2)       # [B, 64, 32, T]

        # 解码器 + 跳跃连接
        d3 = self.dec3(e3)
        d3_cat = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e1], dim=1)

        mask = self.dec1(d2_cat)
        return x * mask
