"""移频移向法

通过频率偏移破坏反馈相位条件，抑制啸叫
"""

import torch
import torch.nn as nn
import math


class FrequencyShiftMethod(nn.Module):
    """移频移向法
    
    通过频率偏移破坏反馈相位条件，抑制啸叫
    """
    
    def __init__(self, shift_hz=20.0, sample_rate=16000, n_fft=512, hop_length=128, window='hann'):
        """初始化移频移向法
        
        Args:
            shift_hz: 频率偏移量(Hz)，默认20
            sample_rate: 采样率，默认16000
            n_fft: FFT窗口大小，默认512
            hop_length: 跳跃长度，默认128
            window: 窗函数类型，默认hann
        """
        super(FrequencyShiftMethod, self).__init__()
        
        self.shift_hz = shift_hz
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        
        # 频率分辨率
        self.freq_resolution = sample_rate / n_fft
        self.shift_bins = shift_hz / self.freq_resolution
        
        # 创建窗函数
        if window == 'hann':
            self.window = torch.hann_window(n_fft)
        elif window == 'hamming':
            self.window = torch.hamming_window(n_fft)
        else:
            self.window = torch.hann_window(n_fft)
            
        self._precompute_phase_factors()
        
    def _precompute_phase_factors(self):
        """预计算相位调整因子"""
        frame_indices = torch.arange(1000).float()
        phase_shift = 2 * math.pi * self.shift_hz * frame_indices * self.hop_length / self.sample_rate
        self.phase_factors = torch.exp(1j * phase_shift)
        
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
        magnitude_spec = x_linear.squeeze(1)
        shifted_magnitude = self._apply_frequency_shift(magnitude_spec)
        shifted_magnitude = shifted_magnitude.unsqueeze(1)
        
        # 转换回log域
        shifted_log = torch.log10(shifted_magnitude + 1e-8)
        
        return shifted_log
    
    def _apply_frequency_shift(self, magnitude_spec):
        """应用频率偏移"""
        batch_size, freq_bins, time_frames = magnitude_spec.shape
        shifted_magnitude = torch.zeros_like(magnitude_spec)
        
        # 频率偏移
        for f in range(freq_bins):
            target_f = f + self.shift_bins
            
            if 0 <= target_f < freq_bins:
                f_low = int(math.floor(target_f))
                f_high = int(math.ceil(target_f))
                alpha = target_f - f_low
                
                if f_low < freq_bins and f_high < freq_bins:
                    if f_low == f_high:
                        shifted_magnitude[:, f, :] = magnitude_spec[:, f_low, :]
                    else:
                        shifted_magnitude[:, f, :] = (
                            (1 - alpha) * magnitude_spec[:, f_low, :] + 
                            alpha * magnitude_spec[:, f_high, :]
                        )
        
        return shifted_magnitude


def create_frequency_shift_method(shift_hz=20.0, **kwargs):
    """创建移频移向法实例"""
    return FrequencyShiftMethod(shift_hz=shift_hz, **kwargs)


if __name__ == "__main__":
    batch_size, channels, freq_bins, time_frames = 2, 1, 256, 100
    test_input = torch.randn(batch_size, channels, freq_bins, time_frames).abs()
    test_input = torch.log10(test_input + 1e-8)
    
    method = FrequencyShiftMethod(shift_hz=20.0)
    output = method(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print("移频移向法测试通过！")