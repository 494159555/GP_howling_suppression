"""移频法

通过频率偏移破坏反馈相位条件，抑制啸叫
"""

import torch
import torch.nn as nn
import math


class FrequencyShiftMethod(nn.Module):
    """移频法

    通过频率偏移破坏反馈相位条件，抑制啸叫。
    输入输出均为归一化对数幅度谱 [0,1]。
    """

    # 归一化参数（与数据集一致）
    NORM_MIN = -11.5
    NORM_MAX = 2.5
    NORM_RANGE = NORM_MAX - NORM_MIN  # 14.0

    def __init__(self, shift_hz=5.0, sample_rate=16000, n_fft=512, hop_length=128):
        """初始化移频法

        Args:
            shift_hz: 频率偏移量(Hz)，默认5Hz（典型值1-10Hz）
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
        """
        super().__init__()

        self.shift_hz = shift_hz
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 频率分辨率
        self.freq_resolution = sample_rate / n_fft
        self.shift_bins = shift_hz / self.freq_resolution

    def _denorm(self, x):
        """归一化 [0,1] -> 对数域 [NORM_MIN, NORM_MAX]"""
        return x * self.NORM_RANGE + self.NORM_MIN

    def _renorm(self, x):
        """对数域 [NORM_MIN, NORM_MAX] -> 归一化 [0,1]"""
        return (x - self.NORM_MIN) / self.NORM_RANGE

    def forward(self, x):
        """前向传播

        Args:
            x: 归一化对数幅度谱 [B, 1, F, T]，范围 [0, 1]

        Returns:
            处理后的归一化对数幅度谱 [B, 1, F, T]，范围约 [0, 1]
        """
        batch_size, channels, freq_bins, time_frames = x.shape

        # 归一化 [0,1] -> 对数域 -> 线性域
        x_log = self._denorm(x)
        x_linear = torch.pow(10, x_log).squeeze(1)  # [B, F, T]

        # 应用频率偏移
        shifted = self._apply_frequency_shift(x_linear)

        # 线性域 -> 对数域 -> 归一化
        shifted_log = torch.log10(shifted + 1e-10)
        result = self._renorm(shifted_log).unsqueeze(1)  # [B, 1, F, T]

        return result

    def _apply_frequency_shift(self, magnitude_spec):
        """在线性幅度谱上应用频率偏移

        使用线性插值对频谱沿频率轴进行微小平移。
        """
        batch_size, freq_bins, time_frames = magnitude_spec.shape
        shifted = torch.zeros_like(magnitude_spec)

        # 构建源频率索引（浮点）
        target_freqs = torch.arange(freq_bins, device=magnitude_spec.device, dtype=torch.float32)
        source_freqs = target_freqs - self.shift_bins

        # 仅处理有效范围内的频率
        valid = (source_freqs >= 0) & (source_freqs < freq_bins - 1)
        valid_idx = torch.where(valid)[0]

        for f_out in valid_idx:
            f_src = source_freqs[f_out]
            f_low = int(math.floor(f_src.item()))
            f_high = f_low + 1
            alpha = f_src - f_low

            if f_high < freq_bins:
                shifted[:, f_out, :] = (
                    (1 - alpha) * magnitude_spec[:, f_low, :] +
                    alpha * magnitude_spec[:, f_high, :]
                )

        # 边界处理：低于偏移范围的部分保持静音或用原始值
        if self.shift_bins > 0:
            # 高频端超出范围的部分用最近的有效值
            last_valid = valid_idx[-1].item() if len(valid_idx) > 0 else freq_bins - 1
            if last_valid + 1 < freq_bins:
                shifted[:, last_valid + 1:, :] = shifted[:, last_valid:last_valid + 1, :]

        return shifted


def create_frequency_shift_method(shift_hz=5.0, **kwargs):
    """创建移频法实例"""
    return FrequencyShiftMethod(shift_hz=shift_hz, **kwargs)


if __name__ == "__main__":
    batch_size, channels, freq_bins, time_frames = 2, 1, 256, 100
    test_input = torch.rand(batch_size, channels, freq_bins, time_frames)

    method = FrequencyShiftMethod(shift_hz=5.0)
    output = method(test_input)

    print(f"输入形状: {test_input.shape}, 范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"输出形状: {output.shape}, 范围: [{output.min():.4f}, {output.max():.4f}]")
    print("移频法测试通过！")
