"""U-Net v6 - 5层U-Net 综合优化版（注意力+残差+空洞）

结合注意力门 + 残差连接 + 空洞卷积，最强版本。
"""

import torch
import torch.nn as nn

from .blocks import make_encoder_block, make_decoder_block, make_output_block
from .modules.attention_modules import AttentionBlock, ResidualBlock, AtrousConvBlock


class AudioUNet5Optimized(nn.Module):
    """综合优化的5层U-Net音频啸叫抑制模型

    三重改进：注意力门 + 残差连接 + 空洞卷积

    输入: [batch, 1, 256, time]
    输出: [batch, 1, 256, time]
    """

    def __init__(self, dilation_rates: list = [2, 4, 8]):
        super(AudioUNet5Optimized, self).__init__()

        # 编码器: 下采样 + 残差块
        self.enc1_down = make_encoder_block(1, 16)
        self.res1 = ResidualBlock(16)

        self.enc2_down = make_encoder_block(16, 32)
        self.res2 = ResidualBlock(32)

        self.enc3_down = make_encoder_block(32, 64)
        self.res3 = ResidualBlock(64)

        self.enc4_down = make_encoder_block(64, 128)
        self.res4 = ResidualBlock(128)

        self.enc5_down = make_encoder_block(128, 256)
        self.res5 = ResidualBlock(256)

        # 瓶颈层空洞卷积
        self.atrous_block = AtrousConvBlock(256, 256, dilation_rates=dilation_rates)

        # 解码器
        self.dec5 = make_decoder_block(256, 128)
        self.dec4 = make_decoder_block(256, 64)
        self.dec3 = make_decoder_block(128, 32)
        self.dec2 = make_decoder_block(64, 16)
        self.dec1 = make_output_block(32)

        # 注意力门
        self.att4 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.att2 = AttentionBlock(F_g=32, F_l=32, F_int=16)
        self.att1 = AttentionBlock(F_g=16, F_l=16, F_int=8)

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)

        # 编码器 (下采样 + 残差)
        e1 = self.res1(self.enc1_down(x_log))
        e2 = self.res2(self.enc2_down(e1))
        e3 = self.res3(self.enc3_down(e2))
        e4 = self.res4(self.enc4_down(e3))
        e5 = self.atrous_block(self.res5(self.enc5_down(e4)))

        # 解码器 + 注意力门
        d5 = self.dec5(e5)
        e4_att = self.att4(d5, e4)
        d5_cat = torch.cat([d5, e4_att], dim=1)

        d4 = self.dec4(d5_cat)
        e3_att = self.att3(d4, e3)
        d4_cat = torch.cat([d4, e3_att], dim=1)

        d3 = self.dec3(d4_cat)
        e2_att = self.att2(d3, e2)
        d3_cat = torch.cat([d3, e2_att], dim=1)

        d2 = self.dec2(d3_cat)
        e1_att = self.att1(d2, e1)
        d2_cat = torch.cat([d2, e1_att], dim=1)

        mask = self.dec1(d2_cat)
        return x * mask
