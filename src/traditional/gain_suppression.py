"""增益抑制法

自适应增益控制，抑制啸叫频段
"""

import torch
import torch.nn as nn
import math


class GainSuppressionMethod(nn.Module):
    """增益抑制法
    
    检测啸叫频段并应用自适应增益衰减
    """
    
    def __init__(self, 
                 threshold_db=-30.0,
                 attack_time=0.01,
                 release_time=0.1,
                 max_attenuation=-20.0,
                 sample_rate=16000,
                 n_fft=512,
                 hop_length=128,
                 min_freq=1000.0,
                 max_freq=8000.0):
        """初始化增益抑制法
        
        Args:
            threshold_db: 检测阈值，默认-30dB
            attack_time: 攻击时间，默认0.01s
            release_time: 释放时间，默认0.1s
            max_attenuation: 最大衰减量，默认-20dB
            sample_rate: 采样率，默认16000
            n_fft: FFT窗口，默认512
            hop_length: 跳跃长度，默认128
            min_freq: 最小检测频率，默认1000Hz
            max_freq: 最大检测频率，默认8000Hz
        """
        super(GainSuppressionMethod, self).__init__()
        
        self.threshold_db = threshold_db
        self.attack_time = attack_time
        self.release_time = release_time
        self.max_attenuation = max_attenuation
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # 频率bin范围
        self.freq_resolution = sample_rate / n_fft
        self.min_bin = int(min_freq / self.freq_resolution)
        self.max_bin = int(max_freq / self.freq_resolution)
        
        # 攻击和释放系数
        frame_rate = sample_rate / hop_length
        self.attack_coeff = math.exp(-1.0 / (attack_time * frame_rate))
        self.release_coeff = math.exp(-1.0 / (release_time * frame_rate))
        
        # 线性域转换
        self.threshold_linear = 10 ** (threshold_db / 20)
        self.max_attenuation_linear = 10 ** (max_attenuation / 20)
        
        self.register_buffer('gain_mask', None)
        self.register_buffer('background_estimate', None)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入频谱图 [B, 1, F, T]
            
        Returns:
            处理后的频谱图 [B, 1, F, T]
        """
        batch_size, channels, freq_bins, time_frames = x.shape
        
        # 转换为线性域
        x_linear = torch.pow(10, x)
        
        # 初始化状态
        if self.gain_mask is None or self.gain_mask.shape[-1] != time_frames:
            self.gain_mask = torch.ones(batch_size, freq_bins, time_frames, device=x.device)
            self.background_estimate = torch.zeros(batch_size, freq_bins, time_frames, device=x.device)
        
        # 处理时间帧
        processed_spec = torch.zeros_like(x_linear.squeeze(1))
        
        for t in range(time_frames):
            current_frame = x_linear[:, 0, :, t]
            
            # 更新背景估计
            if t == 0:
                self.background_estimate[:, :, t] = current_frame
            else:
                alpha = 0.95
                self.background_estimate[:, :, t] = (
                    alpha * self.background_estimate[:, :, t-1] + 
                    (1 - alpha) * current_frame
                )
            
            # 计算信噪比
            snr = current_frame / (self.background_estimate[:, :, t] + 1e-8)
            
            # 检测啸叫
            howling_mask = self._detect_howling(snr, t)
            
            # 更新增益掩码
            if t == 0:
                self.gain_mask[:, :, t] = torch.ones(batch_size, freq_bins, device=x.device)
            else:
                target_gain = torch.where(
                    howling_mask,
                    torch.full_like(howling_mask, self.max_attenuation_linear, dtype=torch.float32),
                    torch.ones_like(howling_mask, dtype=torch.float32)
                )
                
                gain_diff = target_gain - self.gain_mask[:, :, t-1]
                gain_change = torch.where(
                    gain_diff < 0,
                    gain_diff * (1 - self.attack_coeff),
                    gain_diff * (1 - self.release_coeff)
                )
                
                self.gain_mask[:, :, t] = self.gain_mask[:, :, t-1] + gain_change
            
            processed_spec[:, :, t] = current_frame * self.gain_mask[:, :, t]
        
        # 转换回log域
        processed_log = torch.log10(processed_spec.unsqueeze(1) + 1e-8)
        
        return processed_log
    
    def _detect_howling(self, snr, time_frame):
        """检测啸叫频段"""
        batch_size, freq_bins = snr.shape
        howling_mask = torch.zeros_like(snr, dtype=torch.bool)
        
        for b in range(batch_size):
            for f in range(self.min_bin, min(self.max_bin, freq_bins)):
                if snr[b, f] > self.threshold_linear:
                    # 检查局部峰值
                    is_peak = True
                    for df in [-2, -1, 1, 2]:
                        nf = f + df
                        if 0 <= nf < freq_bins and snr[b, nf] >= snr[b, f]:
                            is_peak = False
                            break
                    
                    if is_peak:
                        howling_mask[b, f] = True
        
        return howling_mask


def create_gain_suppression_method(threshold_db=-30.0, **kwargs):
    """创建增益抑制法实例"""
    return GainSuppressionMethod(threshold_db=threshold_db, **kwargs)


if __name__ == "__main__":
    batch_size, channels, freq_bins, time_frames = 2, 1, 256, 100
    test_input = torch.randn(batch_size, channels, freq_bins, time_frames).abs()
    test_input = torch.log10(test_input + 1e-8)
    test_input[:, 0, 50:60, 20:80] += 2.0
    
    method = GainSuppressionMethod(threshold_db=-30.0)
    output = method(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输入范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print("增益抑制法测试通过！")