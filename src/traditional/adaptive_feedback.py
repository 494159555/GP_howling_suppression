"""自适应反馈抵消法

使用自适应滤波器建模并消除反馈路径
"""

import torch
import torch.nn as nn


class AdaptiveFeedbackMethod(nn.Module):
    """自适应反馈抵消法
    
    使用LMS/NLMS算法实时估计并消除反馈路径
    """
    
    def __init__(self, 
                 filter_length=64,
                 step_size=0.01,
                 leakage_factor=0.9999,
                 normalization=True,
                 max_gain=20.0,
                 sample_rate=16000,
                 n_fft=512,
                 hop_length=128):
        """初始化自适应反馈抵消法
        
        Args:
            filter_length: 滤波器长度，默认64
            step_size: LMS步长，默认0.01
            leakage_factor: 泄漏因子，默认0.9999
            normalization: 是否使用NLMS，默认True
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
        
        self.register_buffer('filter_coeffs', None)
        self.register_buffer('input_buffer', None)
        self.register_buffer('output_buffer', None)
        
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
        if (self.filter_coeffs is None or 
            self.filter_coeffs.shape[0] != batch_size or
            self.input_buffer is None or
            self.input_buffer.shape[-1] != time_frames):
            
            self.filter_coeffs = torch.zeros(batch_size, self.filter_length, device=x.device)
            self.input_buffer = torch.zeros(batch_size, time_frames + self.filter_length, device=x.device)
            self.output_buffer = torch.zeros(batch_size, time_frames, device=x.device)
        
        # 处理时间帧
        processed_spec = torch.zeros_like(x_linear.squeeze(1))
        
        for t in range(time_frames):
            current_frame = x_linear[:, 0, :, t]
            input_signal = torch.mean(current_frame, dim=1)
            
            # 更新输入缓冲区
            self.input_buffer[:, t + self.filter_length] = input_signal
            
            # 自适应滤波
            for b in range(batch_size):
                filter_input = self.input_buffer[b, t:t + self.filter_length].flip(0)
                filter_output = torch.dot(self.filter_coeffs[b], filter_input)
                error = input_signal[b] - filter_output
                
                # LMS系数更新
                if self.normalization:
                    input_power = torch.sum(filter_input ** 2) + 1e-8
                    normalized_step = self.step_size / input_power
                else:
                    normalized_step = self.step_size
                
                self.filter_coeffs[b] = (
                    self.leakage_factor * self.filter_coeffs[b] + 
                    normalized_step * error * filter_input
                )
                
                # 限制系数范围
                self.filter_coeffs[b] = torch.clamp(
                    self.filter_coeffs[b], 
                    -self.max_gain_linear, 
                    self.max_gain_linear
                )
                
                self.output_buffer[b, t] = error
            
            # 重建频谱
            for b in range(batch_size):
                gain_factor = torch.abs(self.output_buffer[b, t]) / (torch.abs(input_signal[b]) + 1e-8)
                gain_factor = torch.clamp(gain_factor, 0.1, 2.0)
                processed_spec[b, :, t] = current_frame[b] * gain_factor
        
        # 转换回log域
        processed_log = torch.log10(processed_spec.unsqueeze(1) + 1e-8)
        
        return processed_log
    
    def process_time_domain(self, input_audio):
        """处理时域音频信号"""
        batch_size, channels, samples = input_audio.shape
        
        # 初始化状态
        if (self.filter_coeffs is None or 
            self.filter_coeffs.shape[0] != batch_size):
            
            self.filter_coeffs = torch.zeros(batch_size, self.filter_length, device=input_audio.device)
            self.input_buffer = torch.zeros(batch_size, samples + self.filter_length, device=input_audio.device)
            self.output_buffer = torch.zeros(batch_size, samples, device=input_audio.device)
        
        # 处理样本
        processed_audio = torch.zeros_like(input_audio.squeeze(1))
        
        for t in range(samples):
            for b in range(batch_size):
                input_sample = input_audio[b, 0, t]
                self.input_buffer[b, t + self.filter_length] = input_sample
                
                filter_input = self.input_buffer[b, t:t + self.filter_length].flip(0)
                predicted_feedback = torch.dot(self.filter_coeffs[b], filter_input)
                error_signal = input_sample - predicted_feedback
                
                # 更新滤波器系数
                if self.normalization:
                    input_power = torch.sum(filter_input ** 2) + 1e-8
                    normalized_step = self.step_size / input_power
                else:
                    normalized_step = self.step_size
                
                self.filter_coeffs[b] = (
                    self.leakage_factor * self.filter_coeffs[b] + 
                    normalized_step * error_signal * filter_input
                )
                
                self.filter_coeffs[b] = torch.clamp(
                    self.filter_coeffs[b], 
                    -self.max_gain_linear, 
                    self.max_gain_linear
                )
                
                processed_audio[b, t] = error_signal
        
        return processed_audio.unsqueeze(1)


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