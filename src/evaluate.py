"""模型评估模块

评估训练好的音频啸叫抑制模型，支持：
- 所有5种U-Net模型变体
- 综合评估指标（SNR, PSNR, STOI, 啸叫抑制）
- 传统方法对比
- 可视化报告生成
"""

import argparse
import os
import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import cfg
from src.dataset import HowlingDataset
from src.models import get_model, list_models, MODEL_CLASSES
from src.models.loss_functions import SpectralLoss, MultiTaskLoss


def load_model_from_checkpoint(checkpoint_path: str, model_name: str = None, device: str = None):
    """从检查点加载模型

    支持两种检查点格式：
    1. train.py 保存的完整 checkpoint（含 model_state_dict 等键）
    2. 纯 state_dict

    Args:
        checkpoint_path: 检查点路径
        model_name: 模型名称（从检查点config自动推断，也可手动指定）
        device: 设备

    Returns:
        (model, checkpoint_info)
    """
    if device is None:
        device = cfg.DEVICE

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 判断检查点格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # train.py 保存的完整检查点
        state_dict = checkpoint['model_state_dict']
        ckpt_config = checkpoint.get('config', {})

        # 从检查点配置推断模型名称
        if model_name is None:
            model_class_name = ckpt_config.get('model', '')
            for name, cls in MODEL_CLASSES.items():
                if cls.__name__ == model_class_name:
                    model_name = name
                    break
            if model_name is None:
                model_name = cfg.DEFAULT_MODEL
                print(f"⚠️ 无法从检查点推断模型，使用默认: {model_name}")

        checkpoint_info = {
            'epoch': checkpoint.get('epoch', -1),
            'train_loss': checkpoint.get('train_loss', None),
            'val_loss': checkpoint.get('val_loss', None),
            'best_val_loss': checkpoint.get('best_val_loss', None),
            'config': ckpt_config,
        }
    else:
        # 纯 state_dict
        state_dict = checkpoint
        if model_name is None:
            model_name = cfg.DEFAULT_MODEL
            print(f"⚠️ 检查点为纯state_dict，使用默认模型: {model_name}")
        checkpoint_info = {}

    # 创建并加载模型
    model_class = get_model(model_name)
    model = model_class().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, model_name, checkpoint_info


def evaluate_basic(model, dataloader, criterion, device) -> float:
    """基础评估：计算平均损失

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        平均损失
    """
    total_loss = 0.0

    with torch.no_grad():
        for noisy_mag, clean_mag in dataloader:
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            pred_mag = model(noisy_mag)
            loss = criterion(pred_mag, clean_mag)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def _istft_from_mag_phase(mag_norm, noisy_stft_complex, n_fft, hop_length, norm_min=-11.5, norm_max=2.5):
    """从归一化幅度谱和带噪相位还原时域波形
    
    Args:
        mag_norm: 归一化后的幅度谱 [B, 1, freq_bins, T]
        noisy_stft_complex: 带噪信号的复数STFT [B, 1, freq_bins+1, T] 或 [B, freq_bins+1, T] 或 [freq_bins+1, T]
        n_fft: FFT大小
        hop_length: 跳跃长度
        norm_min: 归一化最小值
        norm_max: 归一化最大值
    
    Returns:
        时域波形 tensor [B, 1, T]（与数据集波形格式一致）
    """
    eps = 1e-8
    # 反归一化
    mag_log = mag_norm * (norm_max - norm_min) + norm_min
    # 反对数
    mag_linear = 10 ** mag_log
    
    # 标准化 noisy_stft_complex 为 3D: [B, freq_bins+1, T]
    # 数据集 complex_stft_transform 输出 [1, 257, T]，DataLoader collate 后为 [B, 1, 257, T]
    if noisy_stft_complex.dim() == 4:
        noisy_stft_complex = noisy_stft_complex.squeeze(1)  # [B, 1, 257, T] → [B, 257, T]
    
    # 获取相位（从noisy_stft_complex）
    phase = torch.angle(noisy_stft_complex)  # [B, 257, T] 或 [257, T]
    
    # 确保维度匹配：mag_norm 是 [B, 1, freq_bins, T]
    # freq_bins = n_fft//2+1 (e.g., 257 for n_fft=512)
    batch_size = mag_norm.shape[0]
    freq_bins = mag_norm.shape[2]
    time_frames = mag_norm.shape[3]
    
    # 对齐相位与幅度谱的频率/时间维度
    if phase.dim() == 3:
        phase = phase[:, :freq_bins, :time_frames]  # [B, freq_bins, T]
    else:
        phase = phase[:freq_bins, :time_frames]
    
    # 去掉channel维度
    mag_flat = mag_linear.squeeze(1)  # [B, freq_bins, T]
    
    # 构造复数STFT [B, freq_bins, T]
    enhanced_stft = mag_flat * torch.exp(1j * phase)
    
    # 补齐Nyquist频率bin：istft要求 freq_bins = n_fft//2+1 (257)
    # 数据集裁剪了最后一帧(257→256)，需要补零恢复
    target_bins = n_fft // 2 + 1
    if enhanced_stft.shape[1] < target_bins:
        pad_bins = target_bins - enhanced_stft.shape[1]
        padding = torch.zeros(
            enhanced_stft.shape[0], pad_bins, enhanced_stft.shape[2],
            dtype=enhanced_stft.dtype, device=enhanced_stft.device
        )
        enhanced_stft_full = torch.cat([enhanced_stft, padding], dim=1)
    else:
        enhanced_stft_full = enhanced_stft
    
    # iSTFT还原
    waveform = torch.istft(enhanced_stft_full, n_fft=n_fft, hop_length=hop_length, 
                           length=int(16000 * 3.0))  # [B, T]
    
    # 统一返回 [B, 1, T] 格式，与数据集波形 (howling_wave/clean_wave) 一致
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(1)  # [B, T] → [B, 1, T]
    
    return waveform


def evaluate_with_metrics(model, dataloader, device, use_timedomain=True) -> Dict[str, float]:
    """综合评估：计算所有指标（含时域指标STOI/PESQ/SI-SDR）

    Args:
        model: 模型
        dataloader: 数据加载器（需使用 return_waveform=True 的数据集）
        device: 设备
        use_timedomain: 是否使用时域评估（STOI/PESQ需要时域信号）

    Returns:
        评估指标字典
    """
    from src.evaluation.metrics import AudioMetrics, calculate_mos_score

    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)

    all_metrics = {
        'snr_improvement_db': [],
        'psnr_db': [],
        'si_sdr_db': [],
        'stoi_score': [],
        'pesq_score': [],
        'howling_reduction_db': [],
        'spectral_smoothness_improvement': [],
        'high_frequency_reduction': [],
    }

    losses = []
    
    # 检测dataloader是否返回波形数据（5个元素 vs 2个元素）
    sample_batch = next(iter(dataloader))
    has_waveform = len(sample_batch) >= 5
    del sample_batch

    with torch.no_grad():
        for batch in dataloader:
            if has_waveform:
                noisy_mag, clean_mag, noisy_wave, clean_wave, noisy_stft = batch
            else:
                noisy_mag, clean_mag = batch
                noisy_wave = clean_wave = noisy_stft = None
            
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            pred_mag = model(noisy_mag)

            if use_timedomain and has_waveform:
                # ★ 时域评估：通过iSTFT还原波形，计算STOI/PESQ/SI-SDR
                try:
                    # 还原增强后的时域波形
                    enhanced_wave = _istft_from_mag_phase(
                        pred_mag.cpu(), noisy_stft, 
                        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH
                    )
                    noisy_wave_td = _istft_from_mag_phase(
                        noisy_mag.cpu(), noisy_stft,
                        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH
                    )
                    clean_wave_td = clean_wave  # 直接使用原始干净波形
                    
                    # 时域信号质量指标（SNR/PSNR/SI-SDR/STOI/PESQ）
                    sample_metrics = {
                        'snr_improvement_db': metrics_calc.calculate_snr(clean_wave_td, enhanced_wave, noisy_wave_td),
                        'psnr_db': metrics_calc.calculate_psnr(clean_wave_td, enhanced_wave),
                        'si_sdr_db': metrics_calc.calculate_si_sdr(clean_wave_td, enhanced_wave),
                        'stoi_score': metrics_calc.calculate_stoi(clean_wave_td, enhanced_wave),
                        'pesq_score': metrics_calc.calculate_pesq(clean_wave_td, enhanced_wave),
                    }
                    # 啸叫抑制指标需要频谱数据（非时域波形）
                    howling_metrics = metrics_calc.calculate_howling_reduction(
                        noisy_mag.cpu(), pred_mag.cpu()
                    )
                    sample_metrics.update(howling_metrics)
                except Exception as e:
                    # 时域评估失败时回退到频域评估
                    warnings.warn(f"时域评估失败，回退到频域评估: {e}")
                    sample_metrics = metrics_calc.calculate_all_metrics(
                        clean=clean_mag,
                        noisy=noisy_mag,
                        enhanced=pred_mag,
                    )
            else:
                # 频域评估（无波形数据时的回退方案）
                sample_metrics = metrics_calc.calculate_all_metrics(
                    clean=clean_mag,
                    noisy=noisy_mag,
                    enhanced=pred_mag,
                )

            for key in all_metrics:
                if key in sample_metrics:
                    all_metrics[key].append(sample_metrics[key])

            # L1 loss
            loss = nn.L1Loss()(pred_mag, clean_mag)
            losses.append(loss.item())

    # 汇总
    results = {}
    results['avg_loss'] = sum(losses) / len(losses) if losses else 0.0

    for key, values in all_metrics.items():
        if values:
            results[key] = sum(values) / len(values)
        else:
            results[key] = 0.0

    # MOS估算
    results['mos_estimate'] = calculate_mos_score(results)

    return results


def evaluate_traditional_methods(dataloader, device) -> Dict[str, Dict[str, float]]:
    """评估传统方法用于对比

    Args:
        dataloader: 数据加载器
        device: 设备

    Returns:
        {method_name: metrics_dict}
    """
    from src.traditional import (
        FrequencyShiftMethod,
        GainSuppressionMethod,
        AdaptiveFeedbackMethod,
    )
    from src.evaluation.metrics import AudioMetrics, calculate_mos_score

    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)

    methods = {
        'FrequencyShift': FrequencyShiftMethod(shift_hz=20.0),
        'GainSuppression': GainSuppressionMethod(threshold_db=-30.0),
        'AdaptiveFeedback': AdaptiveFeedbackMethod(filter_length=64),
    }

    all_results = {}
    
    # 检测dataloader是否返回波形数据
    sample_batch = next(iter(dataloader))
    has_waveform = len(sample_batch) >= 5
    del sample_batch

    for method_name, method in methods.items():
        method = method.to(device)
        method.eval()

        method_losses = []
        method_metrics = {
            'snr_improvement_db': [],
            'psnr_db': [],
            'si_sdr_db': [],
            'stoi_score': [],
            'pesq_score': [],
            'howling_reduction_db': [],
        }

        print(f"  评估传统方法: {method_name}...")

        with torch.no_grad():
            for batch in dataloader:
                if has_waveform:
                    noisy_mag, clean_mag, noisy_wave, clean_wave, noisy_stft = batch
                else:
                    noisy_mag, clean_mag = batch
                    noisy_wave = clean_wave = noisy_stft = None
                
                noisy_mag = noisy_mag.to(device)
                clean_mag = clean_mag.to(device)

                try:
                    pred_mag = method(noisy_mag)

                    # 基本loss
                    loss = nn.L1Loss()(pred_mag, clean_mag)
                    method_losses.append(loss.item())

                    if has_waveform and noisy_stft is not None:
                        # 时域评估
                        try:
                            enhanced_wave = _istft_from_mag_phase(
                                pred_mag.cpu(), noisy_stft,
                                n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH
                            )
                            noisy_wave_td = _istft_from_mag_phase(
                                noisy_mag.cpu(), noisy_stft,
                                n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH
                            )
                            # 时域信号质量指标
                            sample_metrics = {
                                'snr_improvement_db': metrics_calc.calculate_snr(clean_wave, enhanced_wave, noisy_wave_td),
                                'psnr_db': metrics_calc.calculate_psnr(clean_wave, enhanced_wave),
                                'si_sdr_db': metrics_calc.calculate_si_sdr(clean_wave, enhanced_wave),
                                'stoi_score': metrics_calc.calculate_stoi(clean_wave, enhanced_wave),
                                'pesq_score': metrics_calc.calculate_pesq(clean_wave, enhanced_wave),
                            }
                            # 啸叫抑制指标需要频谱数据
                            howling_metrics = metrics_calc.calculate_howling_reduction(
                                noisy_mag.cpu(), pred_mag.cpu()
                            )
                            sample_metrics.update(howling_metrics)
                        except Exception:
                            sample_metrics = metrics_calc.calculate_all_metrics(
                                clean=clean_mag,
                                noisy=noisy_mag,
                                enhanced=pred_mag,
                            )
                    else:
                        # 频域评估（回退方案）
                        sample_metrics = metrics_calc.calculate_all_metrics(
                            clean=clean_mag,
                            noisy=noisy_mag,
                            enhanced=pred_mag,
                        )

                    for key in method_metrics:
                        if key in sample_metrics:
                            method_metrics[key].append(sample_metrics[key])
                except Exception as e:
                    print(f"    ⚠️ {method_name} 处理失败: {e}")
                    continue

        results = {}
        results['avg_loss'] = sum(method_losses) / len(method_losses) if method_losses else float('inf')

        for key, values in method_metrics.items():
            if values:
                results[key] = sum(values) / len(values)
            else:
                results[key] = 0.0

        results['mos_estimate'] = calculate_mos_score(results)

        all_results[method_name] = results

    return all_results


def generate_visualizations(
    model,
    model_name: str,
    dataloader,
    device: str,
    output_dir: str,
    traditional_results: Optional[Dict] = None,
):
    """生成可视化报告

    Args:
        model: 深度学习模型
        model_name: 模型名称
        dataloader: 数据加载器
        device: 设备
        output_dir: 输出目录
        traditional_results: 传统方法评估结果
    """
    from src.evaluation.visualizer import AudioVisualizer

    visualizer = AudioVisualizer(save_dir=output_dir)

    # 获取一个样本
    sample_noisy, sample_clean = next(iter(dataloader))
    sample_noisy = sample_noisy.to(device)
    sample_clean = sample_clean.to(device)

    with torch.no_grad():
        sample_pred = model(sample_noisy)

    # 频谱对比图
    for i in range(min(1, sample_noisy.shape[0])):
        visualizer.plot_spectrogram_comparison(
            clean_spec=sample_clean[i, 0].cpu(),
            noisy_spec=sample_noisy[i, 0].cpu(),
            enhanced_spec=sample_pred[i, 0].cpu(),
            method_name=model_name,
            save_name=f"spectrogram_{model_name}_sample{i}.png",
        )

    # 如果有传统方法结果，生成对比图
    if traditional_results:
        dl_results = evaluate_with_metrics(model, dataloader, device)
        all_results = {model_name: dl_results}
        all_results.update(traditional_results)

        visualizer.plot_metrics_comparison(
            all_results,
            save_name=f"metrics_comparison_{model_name}.png",
        )

        visualizer.plot_radar_chart(
            all_results,
            save_name=f"radar_chart_{model_name}.png",
        )

    print(f"📊 可视化报告已保存至: {output_dir}")


def evaluate_model(
    checkpoint_path: str,
    model_name: str = None,
    batch_size: int = 4,
    full_metrics: bool = False,
    compare_traditional: bool = False,
    visualize: bool = False,
    output_dir: str = None,
):
    """评估模型

    Args:
        checkpoint_path: 模型检查点路径
        model_name: 模型名称（自动推断或手动指定）
        batch_size: 批大小
        full_metrics: 是否计算完整指标
        compare_traditional: 是否对比传统方法
        visualize: 是否生成可视化
        output_dir: 输出目录

    Returns:
        评估结果字典
    """
    device = cfg.DEVICE
    print(f"正在使用设备: {device}")

    # 1. 数据准备
    if not os.path.exists(str(cfg.VAL_CLEAN_DIR)):
        print(f"警告：验证集路径 {cfg.VAL_CLEAN_DIR} 不存在")
        return None

    val_dataset = HowlingDataset(
        clean_dir=cfg.VAL_CLEAN_DIR,
        howling_dir=cfg.VAL_NOISY_DIR,
        sample_rate=cfg.SAMPLE_RATE,
        chunk_len=cfg.CHUNK_LEN,
        n_fft=cfg.N_FFT,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS
    )

    # 2. 加载模型
    model, resolved_model_name, ckpt_info = load_model_from_checkpoint(
        checkpoint_path, model_name, device
    )

    print(f"模型: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params:,}")

    if ckpt_info:
        print(f"检查点 Epoch: {ckpt_info.get('epoch', 'N/A')}")
        if ckpt_info.get('best_val_loss'):
            print(f"训练最佳 Val Loss: {ckpt_info['best_val_loss']:.4f}")

    # 3. 基础评估
    criterion = SpectralLoss()
    avg_loss = evaluate_basic(model, val_loader, criterion, device)

    results = {
        'model_name': resolved_model_name,
        'model_class': model.__class__.__name__,
        'checkpoint': os.path.basename(checkpoint_path),
        'total_params': total_params,
        'avg_spectral_loss': avg_loss,
    }

    print("\n" + "=" * 50)
    print("基础评估结果")
    print("=" * 50)
    print(f"模型: {model.__class__.__name__}")
    print(f"平均频谱损失: {avg_loss:.4f}")

    # 4. 综合指标（使用带波形的数据集用于时域评估）
    if full_metrics:
        print("\n正在计算综合指标（含时域指标SI-SDR/PESQ/STOI）...")
        # 创建带波形返回的数据集用于时域评估
        val_dataset_td = HowlingDataset(
            clean_dir=cfg.VAL_CLEAN_DIR,
            howling_dir=cfg.VAL_NOISY_DIR,
            sample_rate=cfg.SAMPLE_RATE,
            chunk_len=cfg.CHUNK_LEN,
            n_fft=cfg.N_FFT,
            return_waveform=True,
        )
        val_loader_td = DataLoader(
            val_dataset_td, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS
        )
        metrics = evaluate_with_metrics(model, val_loader_td, device)
        results.update(metrics)

        print("\n" + "-" * 50)
        print("综合评估指标")
        print("-" * 50)
        print(f"  SNR改善: {metrics.get('snr_improvement_db', 0):.2f} dB")
        print(f"  SI-SDR: {metrics.get('si_sdr_db', 0):.2f} dB")
        print(f"  PESQ: {metrics.get('pesq_score', 0):.2f}")
        print(f"  STOI: {metrics.get('stoi_score', 0):.4f}")
        print(f"  PSNR: {metrics.get('psnr_db', 0):.2f} dB")
        print(f"  啸叫抑制: {metrics.get('howling_reduction_db', 0):.2f} dB")
        print(f"  MOS估算: {metrics.get('mos_estimate', 0):.2f}")

    # 5. 传统方法对比（使用波形数据集以确保公平对比）
    traditional_results = None
    if compare_traditional:
        print("\n正在评估传统方法...")
        # 如果已有波形数据集就复用，否则创建一个
        if full_metrics and val_loader_td is not None:
            trad_loader = val_loader_td
        else:
            trad_dataset = HowlingDataset(
                clean_dir=cfg.VAL_CLEAN_DIR,
                howling_dir=cfg.VAL_NOISY_DIR,
                sample_rate=cfg.SAMPLE_RATE,
                chunk_len=cfg.CHUNK_LEN,
                n_fft=cfg.N_FFT,
                return_waveform=True,
            )
            trad_loader = DataLoader(
                trad_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS
            )
        traditional_results = evaluate_traditional_methods(trad_loader, device)

        print("\n" + "-" * 50)
        print("传统方法对比")
        print("-" * 50)
        for method_name, method_metrics in traditional_results.items():
            print(f"  {method_name}:")
            print(f"    Loss: {method_metrics.get('avg_loss', 0):.4f}")
            print(f"    MOS: {method_metrics.get('mos_estimate', 0):.2f}")

        results['traditional_comparison'] = traditional_results

    # 6. 可视化
    if visualize:
        if output_dir is None:
            output_dir = str(Path(checkpoint_path).parent.parent / "evaluation")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n正在生成可视化报告...")
        generate_visualizations(
            model, resolved_model_name, val_loader, device,
            output_dir, traditional_results,
        )

    # 7. 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, default=str)
        print(f"\n📄 结果已保存至: {results_path}")

    print("\n" + "=" * 50)
    print("评估完成")
    print("=" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估音频啸叫抑制模型")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list_models(),
        help="模型名称（自动从检查点推断，也可手动指定）"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="批大小"
    )
    parser.add_argument(
        "--full-metrics", action="store_true",
        help="计算完整评估指标（SNR, PSNR, STOI等）"
    )
    parser.add_argument(
        "--compare-traditional", action="store_true",
        help="与传统方法进行对比"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="生成可视化报告"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录（默认为检查点同级evaluation目录）"
    )

    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        batch_size=args.batch_size,
        full_metrics=args.full_metrics,
        compare_traditional=args.compare_traditional,
        visualize=args.visualize,
        output_dir=args.output_dir,
    )
