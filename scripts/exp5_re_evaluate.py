"""实验5: 重新评估已有checkpoint（修复PESQ/SI-SDR缺失问题）

加载已保存的最佳模型权重，用修复后的指标字典重新计算全部指标。

用法:
    python scripts/exp5_re_evaluate.py
    python scripts/exp5_re_evaluate.py --gpu 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import HowlingDataset
from src.models import get_model
from src.evaluation.metrics import AudioMetrics, calculate_mos_score
from src.evaluate import _istft_from_mag_phase

STRATEGIES = ['CosAnnealing', 'Plateau', 'CyclicLR', 'OneCycle', 'WarmupCosine']
CKPT_DIR = PROJECT_ROOT / 'experiments' / 'exp5_training_strategy'


def evaluate_checkpoint(strategy_name, device):
    ckpt_path = CKPT_DIR / strategy_name / 'checkpoints' / f'{strategy_name}_best.pt'
    if not ckpt_path.exists():
        print(f"  [跳过] checkpoint 不存在: {ckpt_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  评估策略: {strategy_name}")
    print(f"{'='*60}")

    # 加载模型
    model = get_model('unet_v6_optimized')().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 数据集
    val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
    val_dataset_td = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR, return_waveform=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    val_loader_td = DataLoader(val_dataset_td, batch_size=8, shuffle=False, num_workers=2)

    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)
    eval_metrics = {
        'snr_improvement_db': [], 'psnr_db': [],
        'si_sdr_db': [], 'stoi_score': [], 'pesq_score': [],
        'howling_reduction_db': [],
    }
    final_l1_losses = []

    td_iter = iter(val_loader_td)
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            final_l1_losses.append(torch.nn.L1Loss()(pred, clean).item())

            try:
                td_batch = next(td_iter)
                td_noisy, td_clean, td_noisy_wave, td_clean_wave, td_noisy_stft = td_batch
                td_noisy = td_noisy.to(device)
                td_pred = model(td_noisy)
                enhanced_wave = _istft_from_mag_phase(td_pred.cpu(), td_noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                noisy_wave_td = _istft_from_mag_phase(td_noisy.cpu(), td_noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                m = {
                    'snr_improvement_db': metrics_calc.calculate_snr(td_clean_wave, enhanced_wave, noisy_wave_td),
                    'psnr_db': metrics_calc.calculate_psnr(td_clean_wave, enhanced_wave),
                    'si_sdr_db': metrics_calc.calculate_si_sdr(td_clean_wave, enhanced_wave),
                    'stoi_score': metrics_calc.calculate_stoi(td_clean_wave, enhanced_wave),
                    'pesq_score': metrics_calc.calculate_pesq(td_clean_wave, enhanced_wave),
                }
                howling_m = metrics_calc.calculate_howling_reduction(td_noisy.cpu(), td_pred.cpu())
                m.update(howling_m)
            except StopIteration:
                break
            except Exception as e:
                print(f"    [警告] 时域评估失败: {e}")
                m = metrics_calc.calculate_all_metrics(clean=clean, noisy=noisy, enhanced=pred)

            for k in eval_metrics:
                if k in m:
                    eval_metrics[k].append(m[k])

    # 组装结果
    results = {
        'strategy_name': strategy_name,
        'model': 'unet_v6_optimized',
        'avg_l1_loss': float(np.mean(final_l1_losses)),
    }
    for k, v in eval_metrics.items():
        results[k] = float(np.mean(v)) if v else 0.0
    results['mos_estimate'] = calculate_mos_score(results)

    print(f"  SNR={results['snr_improvement_db']:.2f} | STOI={results['stoi_score']:.4f} | "
          f"SI-SDR={results['si_sdr_db']:.2f} | PESQ={results['pesq_score']:.2f} | "
          f"MOS={results['mos_estimate']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='实验5: 重新评估checkpoint')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 读取原始训练数据（保留 train_losses 等字段）
    all_results = []
    for strategy in STRATEGIES:
        orig_path = CKPT_DIR / strategy / 'strategy_comparison_results.json'
        orig_data = None
        if orig_path.exists():
            with open(orig_path) as f:
                orig_list = json.load(f)
                if orig_list:
                    orig_data = orig_list[0]

        new_metrics = evaluate_checkpoint(strategy, device)
        if new_metrics is None:
            continue

        # 合并：用新指标覆盖旧的
        if orig_data:
            for k in ['snr_improvement_db', 'psnr_db', 'si_sdr_db', 'stoi_score',
                       'pesq_score', 'howling_reduction_db', 'mos_estimate', 'avg_l1_loss']:
                orig_data[k] = new_metrics[k]
            all_results.append(orig_data)
        else:
            all_results.append(new_metrics)

    # 保存合并结果
    out_path = CKPT_DIR / 'strategy_comparison_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # 打印表格
    print(f"\n{'='*90}")
    print("  实验5 重新评估结果")
    print(f"{'='*90}")
    print(f"{'策略':20s} | {'SNR':>8s} | {'STOI':>7s} | {'SI-SDR':>7s} | {'PESQ':>6s} | {'MOS':>5s}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: x['mos_estimate'], reverse=True):
        print(f"{r['strategy_name']:20s} | {r['snr_improvement_db']:8.2f} | "
              f"{r['stoi_score']:7.4f} | {r['si_sdr_db']:7.2f} | "
              f"{r['pesq_score']:6.2f} | {r['mos_estimate']:5.2f}")

    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
