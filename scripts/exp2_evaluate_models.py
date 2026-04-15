"""实验2: 统一评估所有模型 + 传统方法

对所有训练好的深度学习模型和传统方法进行全面评估对比。
评估指标包括: SNR改善, PSNR, STOI, 啸叫抑制, MOS估算等。

用法:
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --batch-size 4
    python scripts/evaluate_all.py --skip-traditional
    python scripts/evaluate_all.py --output-dir experiments/exp2_eval_all
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import HowlingDataset
from src.models import MODEL_CLASSES, get_model
from src.models.loss_functions import SpectralLoss
from src.evaluation.metrics import AudioMetrics, calculate_mos_score
from torch.utils.data import DataLoader


def find_best_checkpoints():
    """扫描实验目录，为每个模型找到最佳检查点"""
    exp_dir = PROJECT_ROOT / 'experiments'
    checkpoints = {}

    if not exp_dir.exists():
        return checkpoints

    for exp_path in sorted(exp_dir.iterdir()):
        if not exp_path.is_dir():
            continue
        ckpt_path = exp_path / 'checkpoints' / 'best_model.pth'
        config_path = exp_path / 'config.json'

        if not ckpt_path.exists():
            continue

        # 从config.json确定模型类型
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                exp_config = json.load(f)
            model_class_name = exp_config.get('model', '')
            for name, cls in MODEL_CLASSES.items():
                if cls.__name__ == model_class_name:
                    # 保留最新的检查点
                    if name not in checkpoints:
                        checkpoints[name] = {
                            'path': str(ckpt_path),
                            'epoch': exp_config.get('num_epochs', 0),
                            'exp_name': exp_path.name,
                        }
                    break

    return checkpoints


def load_model_from_checkpoint(model_name, checkpoint_path, device):
    """加载模型"""
    model_class = get_model(model_name)
    model = model_class().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# 使用 src.evaluate 中已修复的 _istft_from_mag_phase（处理4D输入、返回[B,1,T]格式）
from src.evaluate import _istft_from_mag_phase


def evaluate_model(model, model_name, dataloader, device):
    """评估单个深度学习模型"""
    from src.evaluate import _istft_from_mag_phase as _istft
    
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
    inference_times = []
    
    # 检测是否返回波形数据
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

            # 测量推理时间
            start = time.time()
            pred_mag = model(noisy_mag)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_times.append(time.time() - start)

            # 损失
            loss = torch.nn.L1Loss()(pred_mag, clean_mag)
            losses.append(loss.item())

            # 综合指标（时域评估）
            if has_waveform and noisy_stft is not None:
                try:
                    enhanced_wave = _istft_from_mag_phase(pred_mag.cpu(), noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                    noisy_wave_td = _istft_from_mag_phase(noisy_mag.cpu(), noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                    # 时域信号质量指标
                    sample_metrics = {
                        'snr_improvement_db': metrics_calc.calculate_snr(clean_wave, enhanced_wave, noisy_wave_td),
                        'psnr_db': metrics_calc.calculate_psnr(clean_wave, enhanced_wave),
                        'si_sdr_db': metrics_calc.calculate_si_sdr(clean_wave, enhanced_wave),
                        'stoi_score': metrics_calc.calculate_stoi(clean_wave, enhanced_wave),
                        'pesq_score': metrics_calc.calculate_pesq(clean_wave, enhanced_wave),
                    }
                    # 啸叫抑制指标需要频谱数据（非时域波形）
                    howling_metrics = metrics_calc.calculate_howling_reduction(noisy_mag.cpu(), pred_mag.cpu())
                    sample_metrics.update(howling_metrics)
                except Exception:
                    sample_metrics = metrics_calc.calculate_all_metrics(
                        clean=clean_mag,
                        noisy=noisy_mag,
                        enhanced=pred_mag,
                    )
            else:
                sample_metrics = metrics_calc.calculate_all_metrics(
                    clean=clean_mag,
                    noisy=noisy_mag,
                    enhanced=pred_mag,
                )
            for key in all_metrics:
                if key in sample_metrics:
                    all_metrics[key].append(sample_metrics[key])

    # 汇总
    results = {'avg_loss': float(np.mean(losses)) if losses else 0.0}
    for key, values in all_metrics.items():
        results[key] = float(np.mean(values)) if values else 0.0

    results['mos_estimate'] = calculate_mos_score(results)
    results['avg_inference_ms'] = float(np.mean(inference_times) * 1000) if inference_times else 0.0
    results['param_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results['model_name'] = model_name

    return results


def evaluate_traditional_methods(dataloader, device):
    """评估所有传统方法"""
    from src.traditional import (
        FrequencyShiftMethod,
        GainSuppressionMethod,
        AdaptiveFeedbackMethod,
    )

    methods = {
        'FrequencyShift': FrequencyShiftMethod(shift_hz=20.0),
        'GainSuppression': GainSuppressionMethod(threshold_db=-30.0),
        'AdaptiveFeedback': AdaptiveFeedbackMethod(filter_length=64),
    }

    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)
    all_results = {}

    for method_name, method in methods.items():
        method = method.to(device)
        method.eval()

        print(f"    评估: {method_name}...")

        method_losses = []
        method_metrics = {
            'snr_improvement_db': [],
            'psnr_db': [],
            'si_sdr_db': [],
            'stoi_score': [],
            'pesq_score': [],
            'howling_reduction_db': [],
        }
        inference_times = []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) >= 5:
                    noisy_mag, clean_mag, noisy_wave, clean_wave, noisy_stft = batch
                else:
                    noisy_mag, clean_mag = batch
                    noisy_wave = clean_wave = noisy_stft = None
                
                noisy_mag = noisy_mag.to(device)
                clean_mag = clean_mag.to(device)

                try:
                    start = time.time()
                    pred_mag = method(noisy_mag)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    inference_times.append(time.time() - start)

                    loss = torch.nn.L1Loss()(pred_mag, clean_mag)
                    method_losses.append(loss.item())

                    # 时域评估（如果有波形数据）
                    if noisy_stft is not None:
                        try:
                            enhanced_wave = _istft_from_mag_phase(pred_mag.cpu(), noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                            noisy_wave_td = _istft_from_mag_phase(noisy_mag.cpu(), noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                            # 时域信号质量指标
                            sample_metrics = {
                                'snr_improvement_db': metrics_calc.calculate_snr(clean_wave, enhanced_wave, noisy_wave_td),
                                'psnr_db': metrics_calc.calculate_psnr(clean_wave, enhanced_wave),
                                'si_sdr_db': metrics_calc.calculate_si_sdr(clean_wave, enhanced_wave),
                                'stoi_score': metrics_calc.calculate_stoi(clean_wave, enhanced_wave),
                                'pesq_score': metrics_calc.calculate_pesq(clean_wave, enhanced_wave),
                            }
                            # 啸叫抑制指标需要频谱数据
                            howling_metrics = metrics_calc.calculate_howling_reduction(noisy_mag.cpu(), pred_mag.cpu())
                            sample_metrics.update(howling_metrics)
                        except Exception:
                            sample_metrics = metrics_calc.calculate_all_metrics(
                                clean=clean_mag, noisy=noisy_mag, enhanced=pred_mag,
                            )
                    else:
                        sample_metrics = metrics_calc.calculate_all_metrics(
                            clean=clean_mag, noisy=noisy_mag, enhanced=pred_mag,
                        )
                    for key in method_metrics:
                        if key in sample_metrics:
                            method_metrics[key].append(sample_metrics[key])
                except Exception as e:
                    print(f"      警告: {method_name} 处理失败: {e}")
                    continue

        results = {'avg_loss': float(np.mean(method_losses)) if method_losses else float('inf')}
        for key, values in method_metrics.items():
            results[key] = float(np.mean(values)) if values else 0.0
        results['mos_estimate'] = calculate_mos_score(results)
        results['avg_inference_ms'] = float(np.mean(inference_times) * 1000) if inference_times else 0.0
        results['param_count'] = sum(p.numel() for p in method.parameters()) if hasattr(method, 'parameters') else 0
        results['model_name'] = method_name

        all_results[method_name] = results

    return all_results


def evaluate_baseline(dataloader, device):
    """评估未处理（带啸叫）信号的基线指标

    计算clean vs noisy（无任何处理）的SI-SDR、PESQ、STOI等指标，
    作为所有方法的性能下界参考。
    """
    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)

    baseline_metrics = {
        'snr_improvement_db': [],
        'psnr_db': [],
        'si_sdr_db': [],
        'stoi_score': [],
        'pesq_score': [],
        'howling_reduction_db': [],
    }

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) >= 5:
                noisy_mag, clean_mag, noisy_wave, clean_wave, noisy_stft = batch
            else:
                noisy_mag, clean_mag = batch
                noisy_wave = clean_wave = noisy_stft = None

            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            # 基线：不经过任何处理，直接用noisy作为enhanced
            if noisy_stft is not None:
                try:
                    noisy_wave_td = _istft_from_mag_phase(noisy_mag.cpu(), noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                    sample_metrics = {
                        'snr_improvement_db': 0.0,  # 未处理，SNR改善为0
                        'si_sdr_db': metrics_calc.calculate_si_sdr(clean_wave, noisy_wave_td),
                        'stoi_score': metrics_calc.calculate_stoi(clean_wave, noisy_wave_td),
                        'pesq_score': metrics_calc.calculate_pesq(clean_wave, noisy_wave_td),
                        'psnr_db': metrics_calc.calculate_psnr(clean_wave, noisy_wave_td),
                    }
                    # 啸叫抑制指标（noisy vs noisy = 无抑制）
                    howling_metrics = metrics_calc.calculate_howling_reduction(noisy_mag.cpu(), noisy_mag.cpu())
                    sample_metrics.update(howling_metrics)
                except Exception:
                    sample_metrics = metrics_calc.calculate_all_metrics(
                        clean=clean_mag, noisy=noisy_mag, enhanced=noisy_mag,
                    )
                    sample_metrics['snr_improvement_db'] = 0.0
            else:
                sample_metrics = metrics_calc.calculate_all_metrics(
                    clean=clean_mag, noisy=noisy_mag, enhanced=noisy_mag,
                )
                sample_metrics['snr_improvement_db'] = 0.0

            for key in baseline_metrics:
                if key in sample_metrics:
                    baseline_metrics[key].append(sample_metrics[key])

    results = {'avg_loss': 0.0}
    for key, values in baseline_metrics.items():
        results[key] = float(np.mean(values)) if values else 0.0
    results['mos_estimate'] = calculate_mos_score(results)
    results['avg_inference_ms'] = 0.0
    results['param_count'] = 0
    results['model_name'] = 'Baseline (No Processing)'

    return results


def main():
    parser = argparse.ArgumentParser(description='实验2: 统一评估所有模型 + 传统方法')
    parser.add_argument('--batch-size', type=int, default=4, help='评估批大小')
    parser.add_argument('--skip-traditional', action='store_true', help='跳过传统方法评估')
    parser.add_argument('--checkpoints', nargs='+', default=None,
                        help='手动指定检查点路径（格式: model_name=path）')
    parser.add_argument('--output-dir', type=str, default='experiments/exp2_eval_all',
                        help='输出目录')
    args = parser.parse_args()

    device = cfg.DEVICE
    print(f"设备: {device}")

    # 准备数据（使用 return_waveform=True 以支持时域评估 STOI/PESQ/SI-SDR）
    print("\n加载验证集（含波形数据，用于时域评估）...")
    val_dataset = HowlingDataset(
        clean_dir=cfg.VAL_CLEAN_DIR,
        howling_dir=cfg.VAL_NOISY_DIR,
        return_waveform=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS
    )
    print(f"验证样本数: {len(val_dataset)}")

    # 查找或使用指定的检查点
    if args.checkpoints:
        checkpoints = {}
        for item in args.checkpoints:
            if '=' in item:
                name, path = item.split('=', 1)
                checkpoints[name] = {'path': path}
            else:
                print(f"  忽略无效参数: {item} (格式应为 model_name=path)")
    else:
        print("\n扫描已训练模型的检查点...")
        checkpoints = find_best_checkpoints()

    print(f"\n找到 {len(checkpoints)} 个模型检查点:")
    for name, info in checkpoints.items():
        print(f"  - {name}: {info['path']}")

    # 创建输出目录
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 评估深度学习模型
    all_results = {}

    for model_name, ckpt_info in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"  评估模型: {model_name}")
        print(f"{'='*60}")

        try:
            model = load_model_from_checkpoint(model_name, ckpt_info['path'], device)
            results = evaluate_model(model, model_name, val_loader, device)
            all_results[model_name] = results

            print(f"    Loss: {results['avg_loss']:.4f}")
            print(f"    SNR改善: {results['snr_improvement_db']:.2f} dB")
            print(f"    PSNR: {results['psnr_db']:.2f} dB")
            print(f"    STOI: {results['stoi_score']:.4f}")
            print(f"    啸叫抑制: {results['howling_reduction_db']:.2f} dB")
            print(f"    MOS估算: {results['mos_estimate']:.2f}")
            print(f"    推理时间: {results['avg_inference_ms']:.2f} ms")
            print(f"    参数量: {results['param_count']:,}")
        except Exception as e:
            print(f"    评估失败: {e}")
            import traceback
            traceback.print_exc()

    # 评估未处理信号基线
    print(f"\n{'='*60}")
    print("  评估未处理信号基线（No Processing Baseline）")
    print(f"{'='*60}")
    baseline_results = evaluate_baseline(val_loader, device)
    all_results['Baseline'] = baseline_results
    print(f"    SI-SDR: {baseline_results['si_sdr_db']:.2f} dB")
    print(f"    PESQ: {baseline_results['pesq_score']:.4f}")
    print(f"    STOI: {baseline_results['stoi_score']:.4f}")

    # 评估传统方法
    if not args.skip_traditional:
        print(f"\n{'='*60}")
        print("  评估传统方法")
        print(f"{'='*60}")
        traditional_results = evaluate_traditional_methods(val_loader, device)
        all_results.update(traditional_results)

    # 保存完整结果
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)

    # 打印对比表格
    print(f"\n{'='*90}")
    print("  评估结果对比")
    print(f"{'='*90}")
    header = (f"{'方法':25s} | {'SI-SDR':>8s} | {'PESQ':>7s} | {'STOI':>7s} | "
              f"{'SNR改善':>8s} | {'啸叫抑制':>8s} | {'MOS':>5s} | {'推理ms':>8s}")
    print(header)
    print("-" * len(header))

    # 按MOS排序
    sorted_results = sorted(all_results.items(), key=lambda x: x[1].get('mos_estimate', 0), reverse=True)
    for name, metrics in sorted_results:
        print(f"{name:25s} | "
              f"{metrics.get('si_sdr_db', 0.0):8.2f} | "
              f"{metrics.get('pesq_score', 0.0):7.4f} | "
              f"{metrics['stoi_score']:7.4f} | "
              f"{metrics['snr_improvement_db']:8.2f} | "
              f"{metrics['howling_reduction_db']:8.2f} | "
              f"{metrics['mos_estimate']:5.2f} | "
              f"{metrics['avg_inference_ms']:8.2f}")

    print(f"\n结果已保存: {results_path}")


if __name__ == '__main__':
    main()
