"""数据增强模块

音频和频谱数据增强技术
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AudioAugmentation:
    """音频域增强
    
    对音频波形应用多种变换
    """
    
    def __init__(
        self,
        noise_levels=(0.001, 0.01, 0.02, 0.05),
        time_stretch_factors=(0.9, 0.95, 1.0, 1.05, 1.1),
        volume_factors=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        max_shift_ratio=0.1,
        prob=0.5
    ):
        """初始化音频增强"""
        self.noise_levels = noise_levels
        self.time_stretch_factors = time_stretch_factors
        self.volume_factors = volume_factors
        self.max_shift_ratio = max_shift_ratio
        self.prob = prob
    
    def __call__(self, waveform):
        """应用随机增强"""
        waveform = waveform.clone()
        
        # 添加高斯噪声
        if random.random() < self.prob:
            noise_level = random.choice(self.noise_levels)
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise
        
        # 音量调整
        if random.random() < self.prob:
            volume_factor = random.choice(self.volume_factors)
            waveform = waveform * volume_factor
        
        # 时间偏移
        if random.random() < self.prob:
            max_shift = int(waveform.shape[1] * self.max_shift_ratio)
            shift = random.randint(-max_shift, max_shift)
            waveform = F.roll(waveform, shifts=shift, dims=1)
        
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform


class SpecAugment:
    """频谱增强 (SpecAugment)
    
    对频谱图应用掩码和扰动
    """
    
    def __init__(
        self,
        freq_mask_param=20,
        time_mask_param=20,
        num_freq_masks=2,
        num_time_masks=2,
        jitter_std=0.01,
        prob=0.5
    ):
        """初始化频谱增强"""
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.jitter_std = jitter_std
        self.prob = prob
    
    def __call__(self, spectrogram):
        """应用随机增强"""
        spec = spectrogram.clone()
        _, freq_bins, time_frames = spec.shape
        
        # 频率掩码
        if random.random() < self.prob:
            for _ in range(self.num_freq_masks):
                mask_width = random.randint(0, self.freq_mask_param)
                mask_start = random.randint(0, freq_bins - mask_width)
                spec[:, mask_start:mask_start + mask_width, :] = 0.0
        
        # 时间掩码
        if random.random() < self.prob:
            for _ in range(self.num_time_masks):
                mask_width = random.randint(0, self.time_mask_param)
                mask_start = random.randint(0, time_frames - mask_width)
                spec[:, :, mask_start:mask_start + mask_width] = 0.0
        
        # 频谱抖动
        if random.random() < self.prob:
            jitter = torch.randn_like(spec) * self.jitter_std
            spec = spec + jitter
        
        spec = torch.clamp(spec, 0.0, 1.0)
        
        return spec


class MixupAugmentation:
    """Mixup增强
    
    样本对的线性插值
    """
    
    def __init__(self, alpha=0.4, prob=0.5):
        """初始化Mixup"""
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, x1, y1, x2, y2):
        """应用Mixup"""
        if random.random() > self.prob:
            return x1, y1, x2, y2
        
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)
        
        mixed_x1 = lam * x1 + (1 - lam) * x2
        mixed_y1 = lam * y1 + (1 - lam) * y2
        
        mixed_x2 = (1 - lam) * x1 + lam * x2
        mixed_y2 = (1 - lam) * y1 + lam * y2
        
        return mixed_x1, mixed_y1, mixed_x2, mixed_y2


class AdversarialAugmentation:
    """对抗训练增强
    
    使用FGSM生成对抗样本
    """
    
    def __init__(self, model, epsilon=0.01, num_steps=1, step_size=0.01, prob=0.3):
        """初始化对抗增强"""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.prob = prob
    
    def __call__(self, x, target):
        """生成对抗样本"""
        if random.random() > self.prob:
            return x
        
        x_adv = x.clone().detach()
        x_adv.requires_grad = True
        
        for _ in range(self.num_steps):
            output = self.model(x_adv)
            loss = F.mse_loss(output, target)
            
            self.model.zero_grad()
            loss.backward()
            
            data_grad = x_adv.grad.data.sign()
            x_adv = x_adv + self.step_size * data_grad
            
            perturbation = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = torch.clamp(x + perturbation, 0.0, 1.0)
            
            x_adv = x_adv.detach()
            x_adv.requires_grad = True
        
        return x_adv


class CombinedAugmentation:
    """组合增强流水线
    
    依次应用多种增强技术
    """
    
    def __init__(
        self,
        use_audio_aug=True,
        use_spec_aug=True,
        use_mixup=False,
        use_adversarial=False,
        model=None
    ):
        """初始化组合增强"""
        self.use_audio_aug = use_audio_aug
        self.use_spec_aug = use_spec_aug
        self.use_mixup = use_mixup
        self.use_adversarial = use_adversarial
        
        self.audio_aug = AudioAugmentation() if use_audio_aug else None
        self.spec_aug = SpecAugment() if use_spec_aug else None
        self.mixup_aug = MixupAugmentation() if use_mixup else None
        
        if use_adversarial and model is not None:
            self.adversarial_aug = AdversarialAugmentation(model)
        else:
            self.adversarial_aug = None
    
    def __call__(self, waveform, spectrogram, target=None):
        """应用组合增强"""
        # 音频增强
        if self.audio_aug is not None:
            aug_waveform = self.audio_aug(waveform)
        else:
            aug_waveform = waveform
        
        # 频谱增强
        if self.spec_aug is not None:
            aug_spec = self.spec_aug(spectrogram)
        else:
            aug_spec = spectrogram
        
        # 对抗增强
        if self.adversarial_aug is not None and target is not None:
            aug_spec = self.adversarial_aug(aug_spec, target)
        
        return aug_waveform, aug_spec, target