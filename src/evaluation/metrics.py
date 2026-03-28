"""
Audio Quality Metrics Module

This module implements various audio quality evaluation metrics including
objective metrics (SNR, PSNR, STOI, PESQ), howling suppression specific metrics,
and computational efficiency metrics.

Author: Research Team
Date: 2026-3-23
Version: 2.0.0
"""

# Standard library imports
import os
import time
from typing import Dict, List, Tuple, Optional

# Third-party imports
import numpy as np
import psutil
import torch
import torchaudio

# Local imports
# None


class AudioMetrics:
    """Audio quality metrics calculator.
    
    This class provides comprehensive audio quality evaluation metrics including
    objective metrics (SNR, PSNR, STOI), howling suppression specific metrics,
    and computational efficiency metrics.
    
    Attributes:
        sample_rate (int): Audio sample rate in Hz
        eps (float): Small value for numerical stability
    """
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize AudioMetrics calculator.
        
        Args:
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
        """
        self.sample_rate = sample_rate
        self.eps = 1e-8
        
    def calculate_snr(self, clean: torch.Tensor, enhanced: torch.Tensor, 
                     noisy: torch.Tensor) -> float:
        """Calculate signal-to-noise ratio (SNR) improvement.
        
        Computes the SNR improvement between input noisy audio and processed enhanced
        audio relative to the clean reference.
        
        Args:
            clean (torch.Tensor): Clean audio reference [B, 1, T]
            enhanced (torch.Tensor): Enhanced/processed audio [B, 1, T]
            noisy (torch.Tensor): Noisy input audio [B, 1, T]
            
        Returns:
            float: SNR improvement in decibels (dB)
        """
        # 计算原始SNR
        snr_input = 10 * torch.log10(
            torch.mean(clean ** 2, dim=-1) / (torch.mean((noisy - clean) ** 2, dim=-1) + self.eps)
        )
        
        # 计算处理后SNR
        snr_output = 10 * torch.log10(
            torch.mean(clean ** 2, dim=-1) / (torch.mean((enhanced - clean) ** 2, dim=-1) + self.eps)
        )
        
        # SNR改善
        snr_improvement = snr_output - snr_input
        
        # 确保返回标量值
        result = snr_improvement.mean()
        if hasattr(result, 'item'):
            return result.item()
        else:
            return float(result)
    
    def calculate_psnr(self, clean: torch.Tensor, enhanced: torch.Tensor) -> float:
        """Calculate peak signal-to-noise ratio (PSNR).
        
        Computes PSNR between clean reference and enhanced audio.
        
        Args:
            clean (torch.Tensor): Clean audio reference
            enhanced (torch.Tensor): Enhanced/processed audio
            
        Returns:
            float: PSNR value in decibels (dB)
        """
        mse = torch.mean((clean - enhanced) ** 2)
        if mse == 0:
            return float('inf')
            
        max_val = torch.max(clean)
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse + self.eps))
        
        # 确保返回标量值
        if hasattr(psnr, 'item'):
            return psnr.item()
        else:
            return float(psnr)
    
    def calculate_stoi(self, clean: torch.Tensor, enhanced: torch.Tensor) -> float:
        """Calculate short-time objective intelligibility (STOI).
        
        Simplified STOI calculation using correlation. For production use,
        consider using the pystoi library for more accurate results.
        
        Args:
            clean (torch.Tensor): Clean audio reference
            enhanced (torch.Tensor): Enhanced/processed audio
            
        Returns:
            float: STOI score between 0 and 1
        """
        # 简化的STOI计算
        # 实际建议使用: from pystoi import stoi
        
        clean_np = clean.detach().cpu().numpy()
        enhanced_np = enhanced.detach().cpu().numpy()
        
        # 计算相关性作为简化指标
        if clean_np.ndim > 1:
            clean_np = clean_np.flatten()
            enhanced_np = enhanced_np.flatten()
            
        correlation = np.corrcoef(clean_np, enhanced_np)[0, 1]
        
        # 确保在0-1范围内
        stoi_score = max(0, min(1, (correlation + 1) / 2))
        
        return float(stoi_score)
    
    def calculate_howling_reduction(self, noisy: torch.Tensor, enhanced: torch.Tensor,
                                  freq_bins: List[int] = None) -> Dict[str, float]:
        """Calculate howling suppression effectiveness metrics.
        
        Computes metrics related to howling reduction including reduction in decibels,
        spectral smoothness improvement, and high frequency energy reduction.
        
        Args:
            noisy (torch.Tensor): Noisy audio spectrogram with howling
            enhanced (torch.Tensor): Enhanced/processed audio spectrogram
            freq_bins (List[int], optional): List of frequency bins corresponding to
                                           howling frequency range. Defaults to None
                                           (uses high frequency bins 200-255).
            
        Returns:
            Dict[str, float]: Dictionary containing howling suppression metrics:
                - howling_reduction_db: Reduction in decibels
                - spectral_smoothness_improvement: Improvement in spectral smoothness
                - high_frequency_reduction: High frequency energy reduction ratio
        """
        if freq_bins is None:
            # 默认高频段作为啸叫频段
            freq_bins = list(range(200, 256))  # 假设512点FFT，256个频bin
            
        # 计算频域能量
        noisy_power = torch.mean(noisy[..., freq_bins, :] ** 2, dim=(-2, -1))
        enhanced_power = torch.mean(enhanced[..., freq_bins, :] ** 2, dim=(-2, -1))
        
        # 啸叫衰减率
        reduction_ratio = 1 - (enhanced_power / (noisy_power + self.eps))
        
        # 确保reduction_ratio在合理范围内
        reduction_ratio = torch.clamp(reduction_ratio, min=0, max=1)
        
        # 频谱平滑度 (简化计算)
        noisy_smoothness = self._calculate_spectral_smoothness(noisy)
        enhanced_smoothness = self._calculate_spectral_smoothness(enhanced)
        
        # 确保返回标量值
        howling_reduction_db = (10 * torch.log10(reduction_ratio + self.eps)).mean()
        spectral_smoothness_improvement = enhanced_smoothness - noisy_smoothness
        high_frequency_reduction = reduction_ratio.mean()
        
        return {
            'howling_reduction_db': howling_reduction_db.item() if hasattr(howling_reduction_db, 'item') else float(howling_reduction_db),
            'spectral_smoothness_improvement': spectral_smoothness_improvement.item() if hasattr(spectral_smoothness_improvement, 'item') else float(spectral_smoothness_improvement),
            'high_frequency_reduction': high_frequency_reduction.item() if hasattr(high_frequency_reduction, 'item') else float(high_frequency_reduction)
        }
    
    def _calculate_spectral_smoothness(self, spectrum: torch.Tensor) -> float:
        """Calculate spectral smoothness metric.
        
        Computes spectral smoothness based on the variance of adjacent frequency bins.
        Higher values indicate smoother spectra (less peakiness).
        
        Args:
            spectrum (torch.Tensor): Input spectrogram
            
        Returns:
            float: Spectral smoothness value (higher is smoother)
        """
        # 计算相邻频bin的差异
        diff = torch.diff(spectrum, dim=-2)
        smoothness = 1 / (torch.mean(diff ** 2) + self.eps)
        result = smoothness
        if hasattr(result, 'item'):
            return result.item()
        else:
            return float(result)
    
    def calculate_computational_metrics(self, method_name: str, input_data: torch.Tensor,
                                      processing_func, **kwargs) -> Dict[str, float]:
        """Calculate computational efficiency metrics.
        
        Measures processing time, memory usage, parameter count, and throughput
        for a given processing method.
        
        Args:
            method_name (str): Name of the processing method
            input_data (torch.Tensor): Input data for processing
            processing_func: Processing function or callable
            **kwargs: Additional arguments for the processing function
            
        Returns:
            Dict[str, float]: Dictionary containing computational metrics:
                - processing_time_ms: Processing time in milliseconds
                - memory_usage_mb: Memory usage in megabytes
                - parameter_count: Number of parameters (if applicable)
                - throughput_samples_per_sec: Samples processed per second
        """
        # 内存使用
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 处理时间
        start_time = time.time()
        
        with torch.no_grad():
            output = processing_func(input_data, **kwargs)
            
        processing_time = time.time() - start_time
        
        # 内存使用后
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # 参数量 (如果是模型)
        param_count = 0
        if hasattr(processing_func, 'parameters'):
            param_count = sum(p.numel() for p in processing_func.parameters())
        
        return {
            'processing_time_ms': processing_time * 1000,
            'memory_usage_mb': memory_usage,
            'parameter_count': param_count,
            'throughput_samples_per_sec': input_data.shape[0] / processing_time if processing_time > 0 else 0
        }
    
    def calculate_all_metrics(self, clean: torch.Tensor, noisy: torch.Tensor, 
                            enhanced: torch.Tensor, method_name: str = "unknown",
                            processing_func=None, **kwargs) -> Dict[str, float]:
        """Calculate all evaluation metrics comprehensively.
        
        Computes audio quality metrics, howling suppression metrics, and
        computational efficiency metrics in one call.
        
        Args:
            clean (torch.Tensor): Clean audio reference
            noisy (torch.Tensor): Noisy input audio with howling
            enhanced (torch.Tensor): Enhanced/processed audio
            method_name (str, optional): Name of the processing method. Defaults to "unknown".
            processing_func (optional): Processing function for computational metrics. Defaults to None.
            **kwargs: Additional arguments for processing function
            
        Returns:
            Dict[str, float]: Dictionary containing all calculated metrics including
                             audio quality, howling suppression, and computational metrics
        """
        metrics = {}
        
        # 音频质量指标
        metrics['snr_improvement_db'] = self.calculate_snr(clean, enhanced, noisy)
        metrics['psnr_db'] = self.calculate_psnr(clean, enhanced)
        metrics['stoi_score'] = self.calculate_stoi(clean, enhanced)
        
        # 啸叫抑制指标
        howling_metrics = self.calculate_howling_reduction(noisy, enhanced)
        metrics.update(howling_metrics)
        
        # 计算效率指标
        if processing_func is not None:
            comp_metrics = self.calculate_computational_metrics(
                method_name, noisy, processing_func, **kwargs
            )
            metrics.update(comp_metrics)
        
        return metrics


def calculate_mos_score(metrics: Dict[str, float]) -> float:
    """Estimate Mean Opinion Score (MOS) from objective metrics.
    
    Uses a simplified weighted formula to estimate MOS (1-5 scale) from
    objective metrics. For production use, consider training a regression
    model with subjective evaluation data.
    
    Args:
        metrics (Dict[str, float]): Dictionary of objective metrics including
                                   snr_improvement_db, stoi_score, psnr_db,
                                   and howling_reduction_db
                                   
    Returns:
        float: Estimated MOS score on a scale of 1 to 5
    """
    # 简化的MOS估算公式
    # 实际应用中建议使用主观评估数据训练回归模型
    
    snr_weight = 0.3
    stoi_weight = 0.4
    psnr_weight = 0.2
    howling_weight = 0.1
    
    # 归一化指标到0-1范围
    snr_norm = min(1.0, max(0.0, metrics.get('snr_improvement_db', 0) / 20))
    stoi_norm = metrics.get('stoi_score', 0)
    psnr_norm = min(1.0, max(0.0, metrics.get('psnr_db', 0) / 40))
    howling_norm = min(1.0, max(0.0, metrics.get('howling_reduction_db', 0) / 10))
    
    # 加权平均
    quality_score = (snr_weight * snr_norm + 
                    stoi_weight * stoi_norm + 
                    psnr_weight * psnr_norm + 
                    howling_weight * howling_norm)
    
    # 映射到MOS 1-5分
    mos_score = 1 + quality_score * 4
    
    return mos_score
