"""U-Net v3 - 5层U-Net + 注意力门机制

在 v2 基础上，所有跳跃连接添加注意力门，自动聚焦啸叫相关特征。
"""

import torch
import torch.nn as nn

from .blocks import make_encoder_block, make_decoder_block, make_output_block
from .modules.attention_modules import AttentionBlock


class AudioUNet5Attention(nn.Module):
    """带注意力门的5层U-Net音频啸叫抑制模型

    输入: [batch, 1, 256, time]
    输出: [batch, 1, 256, time]
    """

    def __init__(self):
        super(AudioUNet5Attention, self).__init__()

        # 编码器: 1→16→32→64→128→256
        self.enc1 = make_encoder_block(1, 16)
        self.enc2 = make_encoder_block(16, 32)
        self.enc3 = make_encoder_block(32, 64)
        self.enc4 = make_encoder_block(64, 128)
        self.enc5 = make_encoder_block(128, 256)

        # 解码器
        self.dec5 = make_decoder_block(256, 128)
        self.dec4 = make_decoder_block(256, 64)
        self.dec3 = make_decoder_block(128, 32)
        self.dec2 = make_decoder_block(64, 16)
        self.dec1 = make_output_block(32)

        # 注意力门: (门控信号通道, 编码器特征通道, 中间通道)
        self.att4 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.att2 = AttentionBlock(F_g=32, F_l=32, F_int=16)
        self.att1 = AttentionBlock(F_g=16, F_l=16, F_int=8)

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)

        # 编码器
        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

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
