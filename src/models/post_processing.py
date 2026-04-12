"""后处理模块

音频啸叫抑制模型的后处理方法，包括：
- 自适应后处理
- 多帧平滑
- 自适应增益控制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class AdaptivePostProcessing:
    """自适应后处理
    
    根据输出特征动态调整降噪阈值和平滑策略
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        threshold_percentile: float = 0.9,
        smoothing_window: int = 5,
        adaptive_threshold: bool = True
    ):
        """初始化自适应后处理
        
        Args:
            threshold: 固定降噪阈值
            threshold_percentile: 自适应阈值的百分位数（0-1）
            smoothing_window: 时间平滑窗口大小
            adaptive_threshold: 是否使用自适应阈值
        """
        self.threshold = threshold
        self.threshold_percentile = threshold_percentile
        self.smoothing_window = smoothing_window
        self.adaptive_threshold = adaptive_threshold
        
        print(f"✅ 自适应后处理已初始化")
        print(f"   固定阈值: {threshold}")
        print(f"   自适应阈值百分位: {threshold_percentile}")
        print(f"   平滑窗口: {smoothing_window}")
        print(f"   自适应模式: {adaptive_threshold}")
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """应用自适应后处理
        
        Args:
            spectrogram: 输入频谱 [batch, channels, freq, time]
            
        Returns:
            处理后的频谱
        """
        # 1. 动态降噪阈值
        if self.adaptive_threshold:
            # 计算每个样本的自适应阈值
            threshold = self._compute_adaptive_threshold(spectrogram)
        else:
            threshold = self.threshold
        
        # 应用阈值掩膜
        mask = (spectrogram > threshold).float()
        spectrogram = spectrogram * mask
        
        # 2. 时间平滑
        spectrogram = self._temporal_smoothing(spectrogram)
        
        # 3. 频率平滑（可选）
        spectrogram = self._frequency_smoothing(spectrogram)
        
        return spectrogram
    
    def _compute_adaptive_threshold(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """计算自适应阈值
        
        Args:
            spectrogram: 输入频谱 [batch, channels, freq, time]
            
        Returns:
            阈值 [batch, channels, 1, 1]
        """
        # 计算每个样本的百分位数
        batch_size, channels, _, _ = spectrogram.shape
        spectrogram_flat = spectrogram.view(batch_size, channels, -1)
        
        # 计算百分位数
        k = int(spectrogram_flat.shape[-1] * self.threshold_percentile)
        sorted_spec, _ = torch.sort(spectrogram_flat, dim=-1)
        threshold = sorted_spec[:, :, k:k+1].view(batch_size, channels, 1, 1)
        
        return threshold
    
    def _temporal_smoothing(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """时间维度的平滑
        
        Args:
            spectrogram: 输入频谱 [batch, channels, freq, time]
            
        Returns:
            平滑后的频谱
        """
        # 转置以在时间维度上应用平均池化
        spec_transposed = spectrogram.permute(0, 1, 3, 2)  # [B, C, T, F]
        
        # 应用平均池化
        smoothed = F.avg_pool1d(
            spec_transposed.reshape(-1, spec_transposed.shape[-2]),
            kernel_size=self.smoothing_window,
            stride=1,
            padding=self.smoothing_window // 2
        )
        
        # 恢复形状
        smoothed = smoothed.reshape(
            spec_transposed.shape[0],
            spec_transposed.shape[1],
            spec_transposed.shape[2],
            spec_transposed.shape[3]
        )
        
        # 转置回原始形状
        smoothed = smoothed.permute(0, 1, 3, 2)  # [B, C, F, T]
        
        return smoothed
    
    def _frequency_smoothing(self, spectrogram: torch.Tensor, window: int = 3) -> torch.Tensor:
        """频率维度的平滑
        
        Args:
            spectrogram: 输入频谱 [batch, channels, freq, time]
            window: 平滑窗口大小
            
        Returns:
            平滑后的频谱
        """
        # 在频率维度上应用平均池化
        padded = F.pad(spectrogram, (0, 0, window//2, window//2), mode='reflect')
        smoothed = F.avg_pool2d(
            padded.permute(0, 1, 3, 2),  # [B, C, T, F]
            kernel_size=(1, window),
            stride=1
        )
        
        return smoothed.permute(0, 1, 3, 2)  # [B, C, F, T]


class MultiFrameSmoother:
    """多帧平滑器
    
    使用多种滤波方法在时间维度上平滑输出
    """
    
    def __init__(
        self,
        method: str = 'moving_average',
        window_size: int = 5,
        kalman_q: float = 0.01,
        kalman_r: float = 0.1
    ):
        """初始化多帧平滑器
        
        Args:
            method: 平滑方法 ('moving_average', 'kalman', 'wiener', 'median')
            window_size: 窗口大小
            kalman_q: Kalman滤波过程噪声
            kalman_r: Kalman滤波观测噪声
        """
        self.method = method
        self.window_size = window_size
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        
        print(f"✅ 多帧平滑器已初始化")
        print(f"   平滑方法: {method}")
        print(f"   窗口大小: {window_size}")
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """应用多帧平滑
        
        Args:
            spectrogram: 输入频谱 [batch, channels, freq, time]
            
        Returns:
            平滑后的频谱
        """
        if self.method == 'moving_average':
            return self._moving_average(spectrogram)
        elif self.method == 'kalman':
            return self._kalman_filter(spectrogram)
        elif self.method == 'wiener':
            return self._wiener_filter(spectrogram)
        elif self.method == 'median':
            return self._median_filter(spectrogram)
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")
    
    def _moving_average(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """移动平均滤波"""
        # 转置到时间维度
        spec_transposed = spectrogram.permute(0, 1, 3, 2)  # [B, C, T, F]
        
        # 应用1D卷积作为移动平均
        kernel = torch.ones(1, 1, self.window_size, device=spectrogram.device) / self.window_size
        kernel = kernel.expand(spectrogram.shape[1], 1, self.window_size)
        
        # 重塑为2D卷积
        spec_reshaped = spec_transposed.reshape(
            spectrogram.shape[0] * spectrogram.shape[1],
            1,
            spectrogram.shape[3],
            spectrogram.shape[2]
        )
        
        # 应用卷积
        padded = F.pad(spec_reshaped, (self.window_size//2, self.window_size//2), mode='reflect')
        smoothed = F.conv1d(padded, kernel, groups=spectrogram.shape[1])
        
        # 恢复形状
        smoothed = smoothed.reshape(spectrogram.shape[0], spectrogram.shape[1], 
                                   spectrogram.shape[3], spectrogram.shape[2])
        smoothed = smoothed.permute(0, 1, 3, 2)
        
        return smoothed
    
    def _kalman_filter(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Kalman滤波
        
        简化的Kalman滤波器实现
        """
        batch_size, channels, freq_bins, time_steps = spectrogram.shape
        result = spectrogram.clone()
        
        # 对每个频率bin独立应用Kalman滤波
        for b in range(batch_size):
            for c in range(channels):
                for f in range(freq_bins):
                    signal = spectrogram[b, c, f, :].cpu().numpy()
                    filtered = self._apply_kalman_1d(signal)
                    result[b, c, f, :] = torch.tensor(filtered, device=spectrogram.device)
        
        return result
    
    def _apply_kalman_1d(self, signal: np.ndarray) -> np.ndarray:
        """对1D信号应用Kalman滤波"""
        n = len(signal)
        x = np.zeros(n)  # 状态估计
        P = np.zeros(n)  # 误差协方差
        
        # 初始化
        x[0] = signal[0]
        P[0] = 1.0
        
        # 预测和更新
        for i in range(1, n):
            # 预测
            x_pred = x[i-1]
            P_pred = P[i-1] + self.kalman_q
            
            # 更新
            K = P_pred / (P_pred + self.kalman_r)  # Kalman增益
            x[i] = x_pred + K * (signal[i] - x_pred)
            P[i] = (1 - K) * P_pred
        
        return x
    
    def _wiener_filter(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """维纳滤波"""
        # 计算功率谱密度
        power_spectrum = spectrogram ** 2
        
        # 估计噪声功率（取最小值作为估计）
        noise_power = torch.min(power_spectrum, dim=-1, keepdim=True)[0]
        
        # 计算信噪比
        snr = power_spectrum / (noise_power + 1e-8)
        
        # 维纳滤波增益
        gain = snr / (snr + 1.0)
        
        # 应用增益
        filtered = spectrogram * gain
        
        return filtered
    
    def _median_filter(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """中值滤波"""
        # 在时间维度上应用中值滤波
        spec_transposed = spectrogram.permute(0, 1, 3, 2)  # [B, C, T, F]
        
        # 使用unfold进行滑动窗口
        batch_size, channels, time_steps, freq_bins = spec_transposed.shape
        spec_unfolded = F.unfold(
            spec_transposed.view(batch_size * channels, 1, time_steps, freq_bins),
            kernel_size=(self.window_size, 1),
            padding=(self.window_size // 2, 0)
        )
        
        # 计算中值
        spec_unfolded = spec_unfolded.view(
            batch_size * channels, freq_bins, self.window_size, -1
        )
        median = torch.median(spec_unfolded, dim=2)[0]
        
        # 恢复形状
        median = median.view(batch_size, channels, freq_bins, time_steps)
        median = median.permute(0, 1, 3, 2)
        
        return median


class AdaptiveGainControl:
    """自适应增益控制
    
    包括自动增益控制（AGC）和动态范围压缩（DRC）
    """
    
    def __init__(
        self,
        method: str = 'agc',
        target_level: float = 0.7,
        compression_ratio: float = 4.0,
        attack_time: float = 0.01,
        release_time: float = 0.1,
        sample_rate: int = 16000
    ):
        """初始化自适应增益控制
        
        Args:
            method: 方法 ('agc', 'drc', 'limiter')
            target_level: 目标电平（0-1）
            compression_ratio: 压缩比
            attack_time: 攻击时间（秒）
            release_time: 释放时间（秒）
            sample_rate: 采样率
        """
        self.method = method
        self.target_level = target_level
        self.compression_ratio = compression_ratio
        self.attack_time = attack_time
        self.release_time = release_time
        self.sample_rate = sample_rate
        
        # 计算平滑系数
        self.attack_coeff = np.exp(-1.0 / (attack_time * sample_rate))
        self.release_coeff = np.exp(-1.0 / (release_time * sample_rate))
        
        print(f"✅ 自适应增益控制已初始化")
        print(f"   方法: {method}")
        print(f"   目标电平: {target_level}")
        if method == 'drc':
            print(f"   压缩比: {compression_ratio}")
            print(f"   攻击时间: {attack_time}s, 释放时间: {release_time}s")
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """应用增益控制
        
        Args:
            spectrogram: 输入频谱 [batch, channels, freq, time]
            
        Returns:
            增益控制后的频谱
        """
        if self.method == 'agc':
            return self._apply_agc(spectrogram)
        elif self.method == 'drc':
            return self._apply_drc(spectrogram)
        elif self.method == 'limiter':
            return self._apply_limiter(spectrogram)
        else:
            raise ValueError(f"Unknown gain control method: {self.method}")
    
    def _apply_agc(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """应用自动增益控制（AGC）"""
        # 计算当前电平（RMS）
        current_level = torch.sqrt(torch.mean(spectrogram ** 2, dim=[2, 3], keepdim=True))
        
        # 计算增益
        gain = self.target_level / (current_level + 1e-8)
        
        # 限制增益范围
        gain = torch.clamp(gain, 0.1, 10.0)
        
        # 应用增益
        result = spectrogram * gain
        
        return result
    
    def _apply_drc(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """应用动态范围压缩（DRC）"""
        # 转换到dB
        spec_db = 20 * torch.log10(torch.abs(spectrogram) + 1e-8)
        
        # 简化的压缩器
        # 计算超过阈值的量
        threshold = 20 * np.log10(self.target_level + 1e-8)
        excess = torch.clamp(spec_db - threshold, 0, None)
        
        # 应用压缩
        spec_db_compressed = spec_db - excess * (1 - 1 / self.compression_ratio)
        
        # 转换回线性域
        result = 10 ** (spec_db_compressed / 20) * torch.sign(spectrogram)
        
        return result
    
    def _apply_limiter(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """应用限幅器"""
        # 计算峰值
        peak = torch.max(torch.abs(spectrogram), dim=-1, keepdim=True)[0]
        peak = torch.max(peak, dim=-2, keepdim=True)[0]
        
        # 计算增益
        gain = torch.ones_like(peak)
        mask = peak > self.target_level
        gain[mask] = self.target_level / peak[mask]
        
        # 应用增益
        result = spectrogram * gain
        
        return result


class PostProcessingPipeline:
    """后处理管道
    
    组合多个后处理方法
    """
    
    def __init__(
        self,
        use_adaptive: bool = True,
        use_smoothing: bool = True,
        use_gain_control: bool = False,
        adaptive_params: dict = None,
        smoothing_params: dict = None,
        gain_params: dict = None
    ):
        """初始化后处理管道
        
        Args:
            use_adaptive: 是否使用自适应后处理
            use_smoothing: 是否使用多帧平滑
            use_gain_control: 是否使用增益控制
            adaptive_params: 自适应后处理参数
            smoothing_params: 平滑参数
            gain_params: 增益控制参数
        """
        self.use_adaptive = use_adaptive
        self.use_smoothing = use_smoothing
        self.use_gain_control = use_gain_control
        
        # 初始化各个后处理模块
        if use_adaptive:
            params = adaptive_params or {}
            self.adaptive_processor = AdaptivePostProcessing(**params)
        else:
            self.adaptive_processor = None
        
        if use_smoothing:
            params = smoothing_params or {}
            self.smoother = MultiFrameSmoother(**params)
        else:
            self.smoother = None
        
        if use_gain_control:
            params = gain_params or {}
            self.gain_controller = AdaptiveGainControl(**params)
        else:
            self.gain_controller = None
        
        print(f"\n✅ 后处理管道已初始化")
        print(f"   自适应处理: {use_adaptive}")
        print(f"   多帧平滑: {use_smoothing}")
        print(f"   增益控制: {use_gain_control}")
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """应用后处理管道
        
        Args:
            spectrogram: 输入频谱 [batch, channels, freq, time]
            
        Returns:
            处理后的频谱
        """
        result = spectrogram
        
        # 按顺序应用后处理方法
        if self.adaptive_processor is not None:
            result = self.adaptive_processor(result)
        
        if self.smoother is not None:
            result = self.smoother(result)
        
        if self.gain_controller is not None:
            result = self.gain_controller(result)
        
        return result
