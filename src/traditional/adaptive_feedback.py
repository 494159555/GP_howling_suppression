"""自适应反馈抵消法

使用自适应滤波器建模并消除反馈路径
"""

import torch
import torch.nn as nn


class AdaptiveFeedbackMethod(nn.Module):
    """自适应反馈抵消法

    使用NLMS算法在频域逐频率bin估计并抑制反馈路径。
    每个频率bin独立维护自适应滤波器增益，通过增益衰减方式抑制反馈，
    同时保证最小增益下限以保留语音可懂度。
    """

    def __init__(self,
                 filter_length=64,
                 step_size=0.1,
                 leakage_factor=0.9999,
                 normalization=True,
                 max_gain=20.0,
                 sample_rate=16000,
                 n_fft=512,
                 hop_length=128):
        """初始化自适应反馈抵消法

        Args:
            filter_length: 滤波器长度（保持API兼容），默认64
            step_size: NLMS步长，默认0.1
            leakage_factor: 泄漏因子，默认0.9999
            normalization: 是否使用NLMS归一化，默认True
            max_gain: 最大增益限制，默认20dB
            sample_rate: 采样率，默认16000
            n_fft: FFT窗口，默认512
            hop_length: 跳跃长度，默认128
        """
        super(AdaptiveFeedbackMethod, self).__init__()

        self.filter_length = filter_length
        self.step_size = step_size
        self.leakage_factor = leakage_factor
        self.normalization = normalization
        self.max_gain = max_gain
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 线性域转换
        self.max_gain_linear = 10 ** (max_gain / 20)

        # 惰性初始化的状态（不使用register_buffer以支持动态batch/频率维度）
        self._fb_gain = None   # [B, F] 每频率bin的反馈增益估计
        self._prev_frame = None  # [B, F] 上一帧输入

    def forward(self, x):
        """前向传播（向量化逐频率bin自适应反馈抵消）

        每个频率bin独立执行NLMS自适应滤波，估计该频率的反馈增益，
        然后通过增益衰减（而非直接减法）抑制反馈成分，保留语音可懂度。

        Args:
            x: 输入频谱图 [B, 1, F, T]

        Returns:
            处理后的频谱图 [B, 1, F, T]
        """
        batch_size, channels, freq_bins, time_frames = x.shape

        # 转换为线性域
        x_linear = torch.pow(10, x).squeeze(1)  # [B, F, T]

        # 初始化每频率bin的自适应状态
        if (self._fb_gain is None or
            self._fb_gain.shape[0] != batch_size or
            self._fb_gain.shape[1] != freq_bins):
            self._fb_gain = torch.full((batch_size, freq_bins), 0.05, device=x.device)
            self._prev_frame = torch.zeros(batch_size, freq_bins, device=x.device)

        processed_spec = torch.zeros_like(x_linear)

        for t in range(time_frames):
            current_frame = x_linear[:, :, t]  # [B, F]

            # 估计反馈分量: fb_gain * prev_frame
            fb_estimate = self._fb_gain * self._prev_frame  # [B, F]

            # 误差信号（当前帧减去估计反馈）
            error = current_frame - fb_estimate

            # NLMS 系数更新
            if self.normalization:
                # 使用全局功率归一化，避免低能量频率bin的数值不稳定
                total_power = torch.mean(self._prev_frame ** 2) + 1e-8
                mu = self.step_size / total_power
            else:
                mu = self.step_size

            self._fb_gain = (
                self.leakage_factor * self._fb_gain +
                mu * error * self._prev_frame
            )

            # 限制反馈增益范围 [0, 0.9]，确保稳定性
            self._fb_gain = torch.clamp(self._fb_gain, 0.0, 0.9)

            # 增益衰减方式抑制反馈（而非直接输出误差信号）
            # fb_gain ∈ [0, 0.9] → suppression_gain ∈ [0.28, 1.0]
            suppression_gain = 1.0 - 0.8 * self._fb_gain

            processed_spec[:, :, t] = current_frame * suppression_gain

            self._prev_frame = current_frame.detach()

        # 转换回log域
        processed_log = torch.log10(processed_spec.unsqueeze(1) + 1e-8)

        return processed_log


def create_adaptive_feedback_method(filter_length=64, **kwargs):
    """创建自适应反馈抵消法实例"""
    return AdaptiveFeedbackMethod(filter_length=filter_length, **kwargs)


if __name__ == "__main__":
    batch_size, channels, freq_bins, time_frames = 2, 1, 256, 100
    test_input = torch.randn(batch_size, channels, freq_bins, time_frames).abs()
    test_input = torch.log10(test_input + 1e-8)
    
    method = AdaptiveFeedbackMethod(filter_length=64, step_size=0.01)
    output = method(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输入范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print("自适应反馈抵消法测试通过！")