"""实验2（改进版）: 统一评估所有模型 + 传统方法（在测试集上）

对所有训练好的深度学习模型和传统方法在测试集上进行全面评估对比。
改进：
- 使用独立的测试集（而非验证集）
- 传统方法使用修复后的实现
- 报告均值和标准差
- 生成论文所需格式的对比表格

用法:
    python scripts/exp2_eval_v2.py
    python scripts/exp2_eval_v2.py --batch-size 4
    python scripts/exp2_eval_v2.py --skip-traditional
    python scripts/exp2_eval_v2.py --output-dir experiments/exp2_eval_v2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import HowlingDataset
from src.models import MODEL_CLASSES, get_model
from src.evaluation.metrics import AudioMetrics, calculate_mos_score
from src.evaluate import _istft_from_mag_phase


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

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                exp_config = json.load(f)
            model_class_name = exp_config.get('model', '')
            for name, cls in MODEL_CLASSES.items():
                if cls.__name__ == model_class_name:
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


def evaluate_single_batch(model_or_method, noisy_mag, device, is_traditional=False):
    """评估单个batch，返回预测频谱"""
    noisy_mag = noisy_mag.to(device)
    with torch.no_grad():
        pred_mag = model_or_method(noisy_mag)
    return pred_mag


def compute_all_metrics_batch(metrics_calc, clean_wave, enhanced_wave, noisy_wave,
                               noisy_mag, pred_mag, clean_mag):
    """计算单个batch的所有指标"""
    sample_metrics = {
        'si_sdr_db': metrics_calc.calculate_si_sdr(clean_wave, enhanced_wave),
        'stoi_score': metrics_calc.calculate_stoi(clean_wave, enhanced_wave),
        'pesq_score': metrics_calc.calculate_pesq(clean_wave, enhanced_wave),
        'snr_improvement_db': metrics_calc.calculate_snr(clean_wave, enhanced_wave, noisy_wave),
    }
    howling_metrics = metrics_calc.calculate_howling_reduction(noisy_mag.cpu(), pred_mag.cpu())
    sample_metrics.update(howling_metrics)

    loss = torch.nn.L1Loss()(pred_mag.cpu(), clean_mag.cpu())
    sample_metrics['loss'] = loss.item()

    return sample_metrics


def evaluate_model_or_method(model_or_method, name, dataloader, device):
    """通用评估函数：评估DL模型或传统方法"""
    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)

    all_metrics = {
        'si_sdr_db': [],
        'stoi_score': [],
        'pesq_score': [],
        'snr_improvement_db': [],
        'howling_reduction_db': [],
        'loss': [],
    }
    inference_times = []

    with torch.no_grad():
        for batch in dataloader:
            noisy_mag, clean_mag, noisy_wave, clean_wave, noisy_stft = batch

            # 推理计时
            start = time.time()
            pred_mag = evaluate_single_batch(model_or_method, noisy_mag, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.time() - start)

            # iSTFT还原时域波形
            try:
                enhanced_wave = _istft_from_mag_phase(
                    pred_mag.cpu(), noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH
                )
                noisy_wave_td = _istft_from_mag_phase(
                    noisy_mag.cpu(), noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH
                )
                sample_metrics = compute_all_metrics_batch(
                    metrics_calc, clean_wave, enhanced_wave, noisy_wave_td,
                    noisy_mag, pred_mag, clean_mag
                )
            except Exception:
                continue

            for key in all_metrics:
                if key in sample_metrics:
                    all_metrics[key].append(sample_metrics[key])

    # 汇总
    results = {}
    for key, values in all_metrics.items():
        if values:
            results[f'{key}_mean'] = float(np.mean(values))
            results[f'{key}_std'] = float(np.std(values))
            results[f'{key}_values'] = [float(v) for v in values]
        else:
            results[f'{key}_mean'] = 0.0
            results[f'{key}_std'] = 0.0
            results[f'{key}_values'] = []

    results['mos_estimate'] = calculate_mos_score(results)
    results['avg_inference_ms'] = float(np.mean(inference_times) * 1000) if inference_times else 0.0
    results['param_count'] = sum(p.numel() for p in model_or_method.parameters()
                                  if p.requires_grad) if hasattr(model_or_method, 'parameters') else 0
    results['model_name'] = name

    return results


def evaluate_baseline(dataloader, device):
    """评估未处理基线"""
    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)

    all_metrics = {
        'si_sdr_db': [],
        'stoi_score': [],
        'pesq_score': [],
    }

    with torch.no_grad():
        for batch in dataloader:
            noisy_mag, clean_mag, noisy_wave, clean_wave, noisy_stft = batch

            try:
                noisy_wave_td = _istft_from_mag_phase(
                    noisy_mag.cpu(), noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH
                )
                all_metrics['si_sdr_db'].append(
                    metrics_calc.calculate_si_sdr(clean_wave, noisy_wave_td)
                )
                all_metrics['stoi_score'].append(
                    metrics_calc.calculate_stoi(clean_wave, noisy_wave_td)
                )
                all_metrics['pesq_score'].append(
                    metrics_calc.calculate_pesq(clean_wave, noisy_wave_td)
                )
            except Exception:
                continue

    results = {}
    for key, values in all_metrics.items():
        if values:
            results[f'{key}_mean'] = float(np.mean(values))
            results[f'{key}_std'] = float(np.std(values))
        else:
            results[f'{key}_mean'] = 0.0
            results[f'{key}_std'] = 0.0

    results['mos_estimate'] = calculate_mos_score(results)
    results['avg_inference_ms'] = 0.0
    results['param_count'] = 0
    results['model_name'] = 'Baseline (No Processing)'

    return results


def main():
    parser = argparse.ArgumentParser(description='实验2（改进版）: 统一评估')
    parser.add_argument('--batch-size', type=int, default=4, help='评估批大小')
    parser.add_argument('--skip-traditional', action='store_true', help='跳过传统方法评估')
    parser.add_argument('--use-val', action='store_true', help='使用验证集（默认使用测试集）')
    parser.add_argument('--output-dir', type=str, default='experiments/exp2_eval_v2',
                        help='输出目录')
    args = parser.parse_args()

    device = cfg.DEVICE
    print(f"设备: {device}")

    # 选择数据集
    if args.use_val:
        clean_dir = cfg.VAL_CLEAN_DIR
        noisy_dir = cfg.VAL_NOISY_DIR
        dataset_name = "验证集"
    else:
        clean_dir = cfg.TEST_CLEAN_DIR
        noisy_dir = cfg.TEST_NOISY_DIR
        dataset_name = "测试集"

    print(f"\n加载{dataset_name}（含波形数据，用于时域评估）...")
    test_dataset = HowlingDataset(
        clean_dir=clean_dir,
        howling_dir=noisy_dir,
        return_waveform=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    print(f"{dataset_name}样本数: {len(test_dataset)}")

    # 查找检查点
    print("\n扫描已训练模型的检查点...")
    checkpoints = find_best_checkpoints()
    print(f"找到 {len(checkpoints)} 个模型检查点:")
    for name, info in checkpoints.items():
        print(f"  - {name}: {info['path']}")

    # 输出目录
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # 1. 评估未处理基线
    print(f"\n{'='*60}")
    print(f"  评估未处理基线")
    print(f"{'='*60}")
    baseline = evaluate_baseline(test_loader, device)
    all_results['Baseline'] = baseline
    print(f"    SI-SDR: {baseline['si_sdr_db_mean']:.2f} ± {baseline['si_sdr_db_std']:.2f} dB")
    print(f"    PESQ: {baseline['pesq_score_mean']:.2f} ± {baseline['pesq_score_std']:.2f}")
    print(f"    STOI: {baseline['stoi_score_mean']:.4f} ± {baseline['stoi_score_std']:.4f}")

    # 2. 评估深度学习模型
    for model_name, ckpt_info in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"  评估模型: {model_name}")
        print(f"{'='*60}")

        try:
            model = load_model_from_checkpoint(model_name, ckpt_info['path'], device)
            results = evaluate_model_or_method(model, model_name, test_loader, device)
            all_results[model_name] = results

            print(f"    SI-SDR: {results['si_sdr_db_mean']:.2f} ± {results['si_sdr_db_std']:.2f} dB")
            print(f"    PESQ: {results['pesq_score_mean']:.2f} ± {results['pesq_score_std']:.2f}")
            print(f"    STOI: {results['stoi_score_mean']:.4f} ± {results['stoi_score_std']:.4f}")
            print(f"    推理: {results['avg_inference_ms']:.2f} ms, 参数: {results['param_count']:,}")
        except Exception as e:
            print(f"    评估失败: {e}")
            import traceback
            traceback.print_exc()

    # 3. 评估传统方法（修复后）
    if not args.skip_traditional:
        print(f"\n{'='*60}")
        print(f"  评估传统方法（修复后实现）")
        print(f"{'='*60}")

        from src.traditional import (
            FrequencyShiftMethod,
            GainSuppressionMethod,
            AdaptiveFeedbackMethod,
        )

        trad_methods = {
            'FrequencyShift': FrequencyShiftMethod(shift_hz=5.0),
            'GainSuppression': GainSuppressionMethod(threshold_db=10.0),
            'AdaptiveFeedback': AdaptiveFeedbackMethod(filter_length=64, step_size=0.05),
        }

        for method_name, method in trad_methods.items():
            method = method.to(device)
            method.eval()
            print(f"    评估: {method_name}...")

            try:
                results = evaluate_model_or_method(method, method_name, test_loader, device)
                all_results[method_name] = results

                print(f"      SI-SDR: {results['si_sdr_db_mean']:.2f} ± {results['si_sdr_db_std']:.2f} dB")
                print(f"      PESQ: {results['pesq_score_mean']:.2f} ± {results['pesq_score_std']:.2f}")
                print(f"      STOI: {results['stoi_score_mean']:.4f} ± {results['stoi_score_std']:.4f}")
            except Exception as e:
                print(f"      评估失败: {e}")

    # 保存结果
    results_path = output_dir / 'evaluation_results_v2.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)

    # 打印论文格式对比表
    print(f"\n{'='*100}")
    print(f"  评估结果对比表（{dataset_name}）")
    print(f"{'='*100}")
    print(f"{'方法':25s} | {'SI-SDR (dB)':>14s} | {'PESQ':>14s} | {'STOI':>14s} | {'参数量':>12s} | {'推理(ms)':>10s}")
    print("-" * 100)

    sorted_results = sorted(all_results.items(),
                            key=lambda x: x[1].get('si_sdr_db_mean', -999), reverse=True)
    for name, m in sorted_results:
        si_sdr = f"{m['si_sdr_db_mean']:.2f}±{m['si_sdr_db_std']:.2f}"
        pesq = f"{m['pesq_score_mean']:.2f}±{m['pesq_score_std']:.2f}"
        stoi = f"{m['stoi_score_mean']:.4f}±{m['stoi_score_std']:.4f}"
        params = f"{m['param_count']:,}"
        infer = f"{m['avg_inference_ms']:.2f}"
        print(f"{name:25s} | {si_sdr:>14s} | {pesq:>14s} | {stoi:>14s} | {params:>12s} | {infer:>10s}")

    print(f"\n结果已保存: {results_path}")


if __name__ == '__main__':
    main()
