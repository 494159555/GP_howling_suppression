"""U-Net v10 - 5层U-Net + GAN生成对抗网络

生成器为5层U-Net，判别器使用 loss_functions.py 中的 Discriminator。
通过对抗训练提高生成质量。
"""

import torch
import torch.nn as nn

from .blocks import make_encoder_block, make_decoder_block, make_output_block
from .loss_functions import Discriminator as _Discriminator


class _Generator(nn.Module):
    """U-Net生成器"""

    def __init__(self):
        super(_Generator, self).__init__()

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

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)

        e1 = self.enc1(x_log)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

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


class AudioUNet5GAN(nn.Module):
    """5层U-Net + GAN框架音频啸叫抑制模型

    generator: U-Net生成器
    discriminator: CNN判别器（来自 loss_functions.Discriminator）

    输入: [batch, 1, 256, time]
    输出: [batch, 1, 256, time]
    """

    def __init__(self):
        super(AudioUNet5GAN, self).__init__()
        self.generator = _Generator()
        self.discriminator = _Discriminator()

    def forward(self, x):
        return self.generator(x)
