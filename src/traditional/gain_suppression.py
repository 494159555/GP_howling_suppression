"""增益抑制法

检测啸叫频段并应用自适应增益衰减
"""

import torch
import torch.nn as nn
import math


class GainSuppressionMethod(nn.Module):
    """增益抑制法

    在对数域检测啸叫频段（远高于背景水平的窄带峰值），
    并对该频段施加增益衰减。输入输出均为归一化对数幅度谱 [0,1]。
    """

    # 归一化参数（与数据集一致）
    NORM_MIN = -11.5
    NORM_MAX = 2.5
    NORM_RANGE = NORM_MAX - NORM_MIN  # 14.0

    def __init__(self,
                 threshold_db=10.0,
                 attack_time=0.01,
                 release_time=0.1,
                 max_attenuation_db=-12.0,
                 sample_rate=16000,
                 n_fft=512,
                 hop_length=128,
                 min_freq=500.0,
                 max_freq=7800.0,
                 smoothing_alpha=0.92):
        """初始化增益抑制法

        Args:
            threshold_db: 啸叫检测阈值(dB)，频谱值超过背景估计多少dB视为啸叫
            attack_time: 攻击时间(s)
            release_time: 释放时间(s)
            max_attenuation_db: 最大衰减量(dB)
            sample_rate: 采样率
            n_fft: FFT窗口
            hop_length: 跳跃长度
            min_freq: 最小检测频率
            max_freq: 最大检测频率
            smoothing_alpha: 背景估计EMA系数
        """
        super().__init__()

        self.threshold_db = threshold_db
        self.max_attenuation_db = max_attenuation_db
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.smoothing_alpha = smoothing_alpha

        # 频率bin范围
        self.freq_resolution = sample_rate / n_fft
        self.min_bin = max(1, int(min_freq / self.freq_resolution))
        self.max_bin = min(int(max_freq / self.freq_resolution), n_fft // 2)

        # 攻击和释放系数
        frame_rate = sample_rate / hop_length
        self.attack_coeff = math.exp(-1.0 / (attack_time * frame_rate))
        self.release_coeff = math.exp(-1.0 / (release_time * frame_rate))

        # 线性域衰减量
        self.max_attenuation_linear = 10 ** (max_attenuation_db / 20)

    def _denorm(self, x):
        """归一化 [0,1] -> 对数域 [NORM_MIN, NORM_MAX]"""
        return x * self.NORM_RANGE + self.NORM_MIN

    def _renorm(self, x):
        """对数域 -> 归一化 [0,1]"""
        return (x - self.NORM_MIN) / self.NORM_RANGE

    def forward(self, x):
        """前向传播

        Args:
            x: 归一化对数幅度谱 [B, 1, F, T]，范围 [0, 1]

        Returns:
            处理后的归一化对数幅度谱 [B, 1, F, T]
        """
        batch_size, channels, freq_bins, time_frames = x.shape

        # 转换到对数域 (dB-like)
        x_log = self._denorm(x).squeeze(1)  # [B, F, T]

        # 在对数域进行背景估计和啸叫检测
        # 背景估计：EMA平滑
        alpha = self.smoothing_alpha
        background = torch.zeros_like(x_log)
        background[:, :, 0] = x_log[:, :, 0]
        for t in range(1, time_frames):
            background[:, :, t] = alpha * background[:, :, t - 1] + (1 - alpha) * x_log[:, :, t]

        # 计算信噪比（对数域差值 = dB差）
        snr_db = x_log - background  # [B, F, T]

        # 局部峰值检测
        is_peak = self._detect_peaks(x_log, freq_bins)

        # 频率范围掩码
        freq_mask = torch.zeros(freq_bins, device=x.device, dtype=torch.bool)
        freq_mask[self.min_bin:self.max_bin] = True

        # 啸叫检测：高于阈值 + 局部峰值 + 在频率范围内
        howling_mask = (snr_db > self.threshold_db) & is_peak & freq_mask.unsqueeze(0).unsqueeze(2)

        # 计算目标增益（对数域：0 或 max_attenuation_db）
        target_gain_db = torch.where(
            howling_mask,
            torch.tensor(self.max_attenuation_db, device=x.device),
            torch.tensor(0.0, device=x.device)
        )

        # 平滑增益变化（攻击/释放）
        smooth_gain_db = torch.zeros_like(target_gain_db)
        smooth_gain_db[:, :, 0] = target_gain_db[:, :, 0]
        for t in range(1, time_frames):
            gain_diff = target_gain_db[:, :, t] - smooth_gain_db[:, :, t - 1]
            coeff = torch.where(
                gain_diff < 0,
                torch.tensor(self.attack_coeff, device=x.device),
                torch.tensor(self.release_coeff, device=x.device)
            )
            smooth_gain_db[:, :, t] = smooth_gain_db[:, :, t - 1] + (1 - coeff) * gain_diff

        # 应用增益（对数域加法 = 线性域乘法）
        # 10^(gain_db/20) 的 log10 = gain_db/20
        processed_log = x_log + smooth_gain_db / 20.0

        # 归一化回 [0,1]
        result = self._renorm(processed_log).unsqueeze(1)

        return result

    def _detect_peaks(self, x_log, freq_bins):
        """检测对数域频谱中的局部峰值

        Args:
            x_log: 对数域幅度谱 [B, F, T]
            freq_bins: 频率bin数量

        Returns:
            is_peak: 峰值掩码 [B, F, T]
        """
        is_peak = torch.ones_like(x_log, dtype=torch.bool)
        for df in [-2, -1, 1, 2]:
            shifted = torch.roll(x_log, shifts=-df, dims=1)
            if df < 0:
                shifted[:, :abs(df), :] = float('-inf')
            else:
                shifted[:, -df:, :] = float('-inf')
            is_peak = is_peak & (x_log > shifted)

        return is_peak


def create_gain_suppression_method(threshold_db=10.0, **kwargs):
    """创建增益抑制法实例"""
    return GainSuppressionMethod(threshold_db=threshold_db, **kwargs)


if __name__ == "__main__":
    batch_size, channels, freq_bins, time_frames = 2, 1, 256, 100
    test_input = torch.rand(batch_size, channels, freq_bins, time_frames)

    # 添加模拟啸叫峰值
    test_input[:, 0, 50:55, 20:80] = 0.9

    method = GainSuppressionMethod(threshold_db=10.0)
    output = method(test_input)

    print(f"输入形状: {test_input.shape}, 范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"输出形状: {output.shape}, 范围: [{output.min():.4f}, {output.max():.4f}]")
    print("增益抑制法测试通过！")
