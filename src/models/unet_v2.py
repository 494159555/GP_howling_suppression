"""U-Net v2 - 5层标准U-Net（默认模型）

编码器5层下采样 + 解码器5层上采样 + 跳跃连接。
参数量约4.5M，项目主力模型。
"""

import torch
import torch.nn as nn

from .blocks import make_encoder_block, make_decoder_block, make_output_block


class AudioUNet5(nn.Module):
    """5层U-Net音频啸叫抑制模型（默认）

    输入: [batch, 1, 256, time]
    输出: [batch, 1, 256, time]
    """

    def __init__(self):
        super(AudioUNet5, self).__init__()

        # 编码器: 1→16→32→64→128→256
        self.enc1 = make_encoder_block(1, 16)
        self.enc2 = make_encoder_block(16, 32)
        self.enc3 = make_encoder_block(32, 64)
        self.enc4 = make_encoder_block(64, 128)
        self.enc5 = make_encoder_block(128, 256)

        # 解码器: 256→128→64→32→16→1 (输入通道含跳跃连接拼接)
        self.dec5 = make_decoder_block(256, 128)
        self.dec4 = make_decoder_block(256, 64)   # 128+128=256
        self.dec3 = make_decoder_block(128, 32)   # 64+64=128
        self.dec2 = make_decoder_block(64, 16)    # 32+32=64
        self.dec1 = make_output_block(32)          # 16+16=32

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)

        # 编码器
        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # 解码器 + 跳跃连接
        d5 = self.dec5(e5)
        d5_cat = torch.cat([d5, e4], dim=1)

        d4 = self.dec4(d5_cat)
        d4_cat = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4_cat)
        d3_cat = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3_cat)
        d2_cat = torch.cat([d2, e1], dim=1)

        mask = self.dec1(d2_cat)
        return x * mask
