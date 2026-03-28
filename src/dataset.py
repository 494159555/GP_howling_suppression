"""音频啸叫抑制数据集模块

实现HowlingDataset类，用于加载和预处理音频数据
"""

import os

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from src.config import cfg


class HowlingDataset(Dataset):
    """音频啸叫抑制数据集
    
    加载成对的干净和啸叫音频文件，支持频谱变换、对数缩放和归一化
    """
    
    def __init__(
        self,
        clean_dir,
        howling_dir,
        sample_rate=None,
        chunk_len=None,
        n_fft=None,
        hop_length=None,
        augment=False,
        audio_aug_params=None,
        spec_aug_params=None
    ):
        self.clean_dir = clean_dir
        self.howling_dir = howling_dir

        # 使用提供的参数或全局配置
        self.sample_rate = sample_rate if sample_rate is not None else cfg.SAMPLE_RATE
        self.chunk_len = chunk_len if chunk_len is not None else cfg.CHUNK_LEN
        self.n_fft = n_fft if n_fft is not None else cfg.N_FFT
        self.hop_length = hop_length if hop_length is not None else cfg.HOP_LENGTH

        # 计算chunk大小
        self.chunk_size = int(self.sample_rate * self.chunk_len)

        # 数据增强设置
        self.augment = augment
        self.audio_aug = None
        self.spec_aug = None
        
        if self.augment:
            from src.models.augmentation import AudioAugmentation, SpecAugment
            
            audio_params = audio_aug_params or {}
            self.audio_aug = AudioAugmentation(**audio_params)
            
            spec_params = spec_aug_params or {}
            self.spec_aug = SpecAugment(**spec_params)

        # 验证目录存在
        if not os.path.exists(str(self.howling_dir)):
            raise FileNotFoundError(f"目录不存在: {self.howling_dir}")

        # 获取排序后的文件名列表
        self.filenames = sorted(os.listdir(str(self.howling_dir)))

        # 初始化频谱变换
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=2.0
        )

    def __len__(self) -> int:
        """数据集样本数"""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (noisy_mag, clean_mag): 预处理后的频谱对，形状[1, 256, T]
        """
        # 1. 加载音频
        file_name = self.filenames[idx]
        howling_path = os.path.join(self.howling_dir, file_name)
        clean_path = os.path.join(self.clean_dir, file_name)

        try:
            howling_wave, sr_h = torchaudio.load(howling_path)
            clean_wave, sr_c = torchaudio.load(clean_path)
        except Exception as e:
            print(f"加载 {file_name} 失败: {e}")
            return self._get_zero_tensors()

        # 2. 音频长度归一化
        if howling_wave.shape[1] < self.chunk_size:
            pad_len = self.chunk_size - howling_wave.shape[1]
            howling_wave = F.pad(howling_wave, (0, pad_len))
            clean_wave = F.pad(clean_wave, (0, pad_len))
        else:
            howling_wave = howling_wave[:, :self.chunk_size]
            clean_wave = clean_wave[:, :self.chunk_size]

        # 3. 应用音频增强
        if self.augment and self.audio_aug is not None:
            howling_wave = self.audio_aug(howling_wave)
            clean_wave = self.audio_aug(clean_wave)

        # 4. 频谱变换
        howling_mag = self.spec_transform(howling_wave).sqrt()
        clean_mag = self.spec_transform(clean_wave).sqrt()

        # 5. 对数变换和归一化
        eps = 1e-8  # 数值稳定性
        
        # 对数变换
        howling_log = torch.log10(howling_mag + eps)
        clean_log = torch.log10(clean_mag + eps)

        # 归一化到[0, 1]
        norm_min = -11.5
        norm_max = 2.5
        howling_norm = (howling_log - norm_min) / (norm_max - norm_min)
        clean_norm = (clean_log - norm_min) / (norm_max - norm_min)

        # 6. 应用频谱增强
        if self.augment and self.spec_aug is not None:
            howling_norm = self.spec_aug(howling_norm)
            clean_norm = self.spec_aug(clean_norm)

        # 7. 调整维度适配U-Net
        # 裁剪最后一帧 (257 -> 256)
        howling_out = howling_norm[:, :-1, :]
        clean_out = clean_norm[:, :-1, :]

        return howling_out, clean_out

    def _get_zero_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """生成零张量作为后备"""
        freq_bins = self.n_fft // 2 + 1
        time_frames = int(self.chunk_size / (self.n_fft / 2)) + 1
        
        freq_bins_cropped = freq_bins - 1
        
        return (
            torch.zeros(1, freq_bins_cropped, time_frames),
            torch.zeros(1, freq_bins_cropped, time_frames)
        )