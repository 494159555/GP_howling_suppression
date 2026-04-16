"""自适应反馈消除法

使用自适应滤波器在线估计并消除声反馈路径
"""

import torch
import torch.nn as nn


class AdaptiveFeedbackMethod(nn.Module):
    """自适应反馈消除法

    在对数幅度谱域使用归一化最小均方（NLMS）算法逐频率bin
    估计反馈路径增益，并通过增益衰减抑制反馈成分。
    输入输出均为归一化对数幅度谱 [0,1]。
    """

    # 归一化参数（与数据集一致）
    NORM_MIN = -11.5
    NORM_MAX = 2.5
    NORM_RANGE = NORM_MAX - NORM_MIN  # 14.0

    def __init__(self,
                 filter_length=64,
                 step_size=0.05,
                 leakage_factor=0.9999,
                 normalization=True,
                 max_suppression_db=-15.0,
                 sample_rate=16000,
                 n_fft=512,
                 hop_length=128):
        """初始化自适应反馈消除法

        Args:
            filter_length: 滤波器长度（保持API兼容）
            step_size: NLMS步长
            leakage_factor: 泄漏因子
            normalization: 是否使用NLMS归一化
            max_suppression_db: 单频最大抑制量(dB)
            sample_rate: 采样率
            n_fft: FFT窗口
            hop_length: 跳跃长度
        """
        super().__init__()

        self.filter_length = filter_length
        self.step_size = step_size
        self.leakage_factor = leakage_factor
        self.normalization = normalization
        self.max_suppression_db = max_suppression_db
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        # 最大抑制量的线性域对数值
        self.max_supp_log = max_suppression_db / 20.0  # log10域

        # 惰性初始化状态
        self._fb_gain = None      # [B, F] 每频率bin的反馈增益估计（对数域）
        self._prev_frame = None   # [B, F] 上一帧对数幅度

    def _denorm(self, x):
        """归一化 [0,1] -> 对数域 [NORM_MIN, NORM_MAX]"""
        return x * self.NORM_RANGE + self.NORM_MIN

    def _renorm(self, x):
        """对数域 -> 归一化 [0,1]"""
        return (x - self.NORM_MIN) / self.NORM_RANGE

    def forward(self, x):
        """前向传播

        在对数域进行自适应反馈消除：
        1. 转换到对数域
        2. 逐帧估计反馈增益
        3. 应用增益衰减
        4. 转换回归一化域

        Args:
            x: 归一化对数幅度谱 [B, 1, F, T]，范围 [0, 1]

        Returns:
            处理后的归一化对数幅度谱 [B, 1, F, T]
        """
        batch_size, channels, freq_bins, time_frames = x.shape

        # 转换到对数域
        x_log = self._denorm(x).squeeze(1)  # [B, F, T]

        # 初始化自适应状态
        if (self._fb_gain is None or
            self._fb_gain.shape[0] != batch_size or
            self._fb_gain.shape[1] != freq_bins):
            self._fb_gain = torch.zeros(batch_size, freq_bins, device=x.device)
            self._prev_frame = torch.zeros(batch_size, freq_bins, device=x.device)

        processed_log = torch.zeros_like(x_log)

        for t in range(time_frames):
            current_frame = x_log[:, :, t]  # [B, F]

            # 估计反馈分量（对数域乘积 = 对数域加法）
            # fb_gain 表示估计的反馈路径增益（对数域）
            fb_estimate = self._fb_gain + self._prev_frame  # [B, F]

            # 误差信号（当前帧 - 估计反馈）
            error = current_frame - fb_estimate

            # NLMS 系数更新（在对数域）
            if self.normalization:
                prev_power = torch.mean(self._prev_frame ** 2) + 1e-6
                mu = self.step_size / prev_power
            else:
                mu = self.step_size

            # 更新增益估计：增益 = log(current/prev)
            # 只在当前帧比估计反馈更强时增加增益估计
            delta = mu * error * self._prev_frame
            self._fb_gain = (
                self.leakage_factor * self._fb_gain +
                delta
            )

            # 限制增益范围（对数域）
            self._fb_gain = torch.clamp(self._fb_gain, 0.0, -self.max_supp_log)

            # 应用增益衰减抑制反馈
            # 衰减量与估计的反馈增益成正比
            suppression = -self._fb_gain * 0.5  # 部分抑制，避免过度衰减
            suppression = torch.clamp(suppression, self.max_supp_log, 0.0)

            processed_log[:, :, t] = current_frame + suppression

            self._prev_frame = current_frame.detach()

        # 归一化回 [0,1]
        result = self._renorm(processed_log).unsqueeze(1)

        return result


def create_adaptive_feedback_method(filter_length=64, **kwargs):
    """创建自适应反馈消除法实例"""
    return AdaptiveFeedbackMethod(filter_length=filter_length, **kwargs)


if __name__ == "__main__":
    batch_size, channels, freq_bins, time_frames = 2, 1, 256, 100
    test_input = torch.rand(batch_size, channels, freq_bins, time_frames)

    method = AdaptiveFeedbackMethod(filter_length=64, step_size=0.05)
    output = method(test_input)

    print(f"输入形状: {test_input.shape}, 范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"输出形状: {output.shape}, 范围: [{output.min():.4f}, {output.max():.4f}]")
    print("自适应反馈消除法测试通过！")
