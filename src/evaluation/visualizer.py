"""
Audio Visualization Module

This module provides comprehensive visualization tools for audio evaluation results,
including spectrogram comparisons, waveform comparisons, metric comparison charts,
radar charts, and comprehensive evaluation reports.

Author: Research Team
Date: 2026-3-23
Version: 2.0.0
"""

# Standard library imports
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Third-party imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec

# Local imports
# None


class AudioVisualizer:
    """Audio evaluation result visualization class.
    
    This class provides comprehensive visualization tools for audio evaluation
    results including spectrograms, waveforms, metric comparisons, radar charts,
    and comprehensive reports.
    
    Attributes:
        save_dir (Path): Directory for saving visualization outputs
        colors (List[str]): Color palette for visualizations
    """
    
    def __init__(self, save_dir: str = "evaluation_results"):
        """Initialize AudioVisualizer.
        
        Args:
            save_dir (str, optional): Directory path for saving visualizations. 
                                    Defaults to "evaluation_results".
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置样式
        sns.set_style("whitegrid")
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
    def plot_spectrogram_comparison(self, clean_spec: torch.Tensor, 
                                  noisy_spec: torch.Tensor, 
                                  enhanced_spec: torch.Tensor,
                                  method_name: str = "Method",
                                  save_name: str = None) -> str:
        """
        绘制频谱图对比
        
        Args:
            clean_spec: 纯净音频频谱 [F, T]
            noisy_spec: 带噪音频频谱 [F, T]
            enhanced_spec: 处理后音频频谱 [F, T]
            method_name: 方法名称
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        if save_name is None:
            save_name = f"spectrogram_comparison_{method_name}.png"
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 转换为numpy
        clean_np = clean_spec.detach().cpu().numpy()
        noisy_np = noisy_spec.detach().cpu().numpy()
        enhanced_np = enhanced_spec.detach().cpu().numpy()
        
        # 绘制频谱图
        im1 = axes[0].imshow(clean_np, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('纯净音频频谱', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('频率bin')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        im2 = axes[1].imshow(noisy_np, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('带噪音频频谱', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('频率bin')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        im3 = axes[2].imshow(enhanced_np, aspect='auto', origin='lower', cmap='viridis')
        axes[2].set_title(f'{method_name} 处理后频谱', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('频率bin')
        axes[2].set_xlabel('时间帧')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_waveform_comparison(self, clean_wave: torch.Tensor,
                               noisy_wave: torch.Tensor,
                               enhanced_wave: torch.Tensor,
                               method_name: str = "Method",
                               sample_rate: int = 16000,
                               save_name: str = None) -> str:
        """
        绘制波形图对比
        
        Args:
            clean_wave: 纯净音频波形 [T]
            noisy_wave: 带噪音频波形 [T]
            enhanced_wave: 处理后音频波形 [T]
            method_name: 方法名称
            sample_rate: 采样率
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        if save_name is None:
            save_name = f"waveform_comparison_{method_name}.png"
            
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # 时间轴
        time_axis = np.arange(len(clean_wave)) / sample_rate
        
        # 绘制波形
        axes[0].plot(time_axis, clean_wave.detach().cpu().numpy(), color='green', linewidth=0.5)
        axes[0].set_title('纯净音频波形', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('幅度')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time_axis, noisy_wave.detach().cpu().numpy(), color='red', linewidth=0.5)
        axes[1].set_title('带噪音频波形', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('幅度')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(time_axis, enhanced_wave.detach().cpu().numpy(), color='blue', linewidth=0.5)
        axes[2].set_title(f'{method_name} 处理后波形', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('幅度')
        axes[2].set_xlabel('时间 (秒)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_metrics_comparison(self, results_dict: Dict[str, Dict[str, float]],
                              save_name: str = "metrics_comparison.png") -> str:
        """
        绘制多方法指标对比图
        
        Args:
            results_dict: 方法结果字典 {method_name: {metric: value}}
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        # 准备数据
        methods = list(results_dict.keys())
        metrics = ['snr_improvement_db', 'psnr_db', 'stoi_score', 'howling_reduction_db']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metric_names = {
            'snr_improvement_db': 'SNR改善 (dB)',
            'psnr_db': 'PSNR (dB)', 
            'stoi_score': 'STOI分数',
            'howling_reduction_db': '啸叫抑制 (dB)'
        }
        
        for i, metric in enumerate(metrics):
            values = [results_dict[method].get(metric, 0) for method in methods]
            
            bars = axes[i].bar(methods, values, color=self.colors[:len(methods)])
            axes[i].set_title(metric_names[metric], fontsize=14, fontweight='bold')
            axes[i].set_ylabel('数值')
            axes[i].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom')
            
            # 旋转x轴标签
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_radar_chart(self, results_dict: Dict[str, Dict[str, float]],
                        save_name: str = "radar_chart.png") -> str:
        """
        绘制雷达图对比
        
        Args:
            results_dict: 方法结果字典
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        # 准备数据
        methods = list(results_dict.keys())
        metrics = ['snr_improvement_db', 'psnr_db', 'stoi_score', 'howling_reduction_db']
        
        # 归一化数据到0-1范围
        normalized_data = {}
        for method in methods:
            normalized_data[method] = []
            for metric in metrics:
                value = results_dict[method].get(metric, 0)
                # 简单归一化
                if metric == 'snr_improvement_db':
                    norm_val = min(1.0, max(0.0, value / 20))
                elif metric == 'psnr_db':
                    norm_val = min(1.0, max(0.0, value / 40))
                elif metric == 'stoi_score':
                    norm_val = min(1.0, max(0.0, value))
                elif metric == 'howling_reduction_db':
                    norm_val = min(1.0, max(0.0, value / 10))
                else:
                    norm_val = 0
                normalized_data[method].append(norm_val)
        
        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        metric_names = ['SNR改善', 'PSNR', 'STOI', '啸叫抑制']
        metric_names += metric_names[:1]  # 闭合图形
        
        # 绘制每个方法
        for i, method in enumerate(methods):
            values = normalized_data[method]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=method, color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.25, color=self.colors[i % len(self.colors)])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('多方法性能对比雷达图', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_computational_comparison(self, results_dict: Dict[str, Dict[str, float]],
                                     save_name: str = "computational_comparison.png") -> str:
        """
        绘制计算效率对比图
        
        Args:
            results_dict: 方法结果字典
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        methods = list(results_dict.keys())
        
        # 计算效率指标
        time_metrics = [results_dict[method].get('processing_time_ms', 0) for method in methods]
        memory_metrics = [results_dict[method].get('memory_usage_mb', 0) for method in methods]
        param_metrics = [results_dict[method].get('parameter_count', 0) for method in methods]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 处理时间
        bars1 = axes[0].bar(methods, time_metrics, color=self.colors[:len(methods)])
        axes[0].set_title('处理时间对比', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('时间 (ms)')
        axes[0].grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, time_metrics):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom')
        
        # 内存使用
        bars2 = axes[1].bar(methods, memory_metrics, color=self.colors[:len(methods)])
        axes[1].set_title('内存使用对比', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('内存 (MB)')
        axes[1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, memory_metrics):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom')
        
        # 参数量
        bars3 = axes[2].bar(methods, param_metrics, color=self.colors[:len(methods)])
        axes[2].set_title('参数量对比', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('参数量')
        axes[2].grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, param_metrics):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:,}', ha='center', va='bottom')
        
        # 旋转x轴标签
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_comprehensive_report(self, results_dict: Dict[str, Dict[str, float]],
                                     save_name: str = "comprehensive_report.png") -> str:
        """
        生成综合评估报告图
        
        Args:
            results_dict: 方法结果字典
            save_name: 保存文件名
            
        Returns:
            保存路径
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. 主要指标对比 (左上)
        ax1 = fig.add_subplot(gs[0, :2])
        methods = list(results_dict.keys())
        metrics = ['snr_improvement_db', 'psnr_db', 'stoi_score']
        
        x = np.arange(len(methods))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results_dict[method].get(metric, 0) for method in methods]
            ax1.bar(x + i*width, values, width, label=metric, 
                   color=self.colors[i % len(self.colors)])
        
        ax1.set_xlabel('方法')
        ax1.set_ylabel('数值')
        ax1.set_title('主要音频质量指标对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(methods, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 雷达图 (右上)
        ax2 = fig.add_subplot(gs[0, 2], projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, method in enumerate(methods):
            values = []
            for metric in metrics:
                value = results_dict[method].get(metric, 0)
                # 归一化
                if metric == 'snr_improvement_db':
                    norm_val = min(1.0, max(0.0, value / 20))
                elif metric == 'psnr_db':
                    norm_val = min(1.0, max(0.0, value / 40))
                else:  # stoi_score
                    norm_val = min(1.0, max(0.0, value))
                values.append(norm_val)
            values += values[:1]
            
            ax2.plot(angles, values, 'o-', linewidth=2, label=method,
                    color=self.colors[i % len(self.colors)])
            ax2.fill(angles, values, alpha=0.25, color=self.colors[i % len(self.colors)])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(['SNR改善', 'PSNR', 'STOI'])
        ax2.set_ylim(0, 1)
        ax2.set_title('性能雷达图', fontsize=12, fontweight='bold')
        ax2.legend()
        
        # 3. 计算效率对比 (中下)
        ax3 = fig.add_subplot(gs[1, :])
        time_metrics = [results_dict[method].get('processing_time_ms', 0) for method in methods]
        memory_metrics = [results_dict[method].get('memory_usage_mb', 0) for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax3.bar(x - width/2, time_metrics, width, label='处理时间 (ms)', 
               color=self.colors[0])
        ax3.bar(x + width/2, memory_metrics, width, label='内存使用 (MB)', 
               color=self.colors[1])
        
        ax3.set_xlabel('方法')
        ax3.set_ylabel('数值')
        ax3.set_title('计算效率对比', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(methods, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 啸叫抑制效果 (左下)
        ax4 = fig.add_subplot(gs[2, 0])
        howling_metrics = [results_dict[method].get('howling_reduction_db', 0) for method in methods]
        
        bars = ax4.bar(methods, howling_metrics, color=self.colors[:len(methods)])
        ax4.set_title('啸叫抑制效果', fontsize=12, fontweight='bold')
        ax4.set_ylabel('抑制量 (dB)')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, howling_metrics):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 5. MOS分数估算 (中下)
        ax5 = fig.add_subplot(gs[2, 1])
        from .metrics import calculate_mos_score
        
        mos_scores = [calculate_mos_score(results_dict[method]) for method in methods]
        
        bars = ax5.bar(methods, mos_scores, color=self.colors[:len(methods)])
        ax5.set_title('MOS分数估算', fontsize=12, fontweight='bold')
        ax5.set_ylabel('MOS分数 (1-5)')
        ax5.set_ylim(1, 5)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, mos_scores):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 6. 综合评分 (右下)
        ax6 = fig.add_subplot(gs[2, 2])
        
        # 计算综合评分
        quality_scores = []
        for method in methods:
            # 归一化各项指标
            snr_norm = min(1.0, max(0.0, results_dict[method].get('snr_improvement_db', 0) / 20))
            psnr_norm = min(1.0, max(0.0, results_dict[method].get('psnr_db', 0) / 40))
            stoi_norm = min(1.0, max(0.0, results_dict[method].get('stoi_score', 0)))
            howling_norm = min(1.0, max(0.0, results_dict[method].get('howling_reduction_db', 0) / 10))
            
            # 加权平均
            score = 0.3 * snr_norm + 0.3 * psnr_norm + 0.2 * stoi_norm + 0.2 * howling_norm
            quality_scores.append(score)
        
        bars = ax6.bar(methods, quality_scores, color=self.colors[:len(methods)])
        ax6.set_title('综合评分', fontsize=12, fontweight='bold')
        ax6.set_ylabel('评分 (0-1)')
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, quality_scores):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle('音频啸叫抑制方法综合评估报告', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
