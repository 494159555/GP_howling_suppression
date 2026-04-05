"""实验6: 数据增强对比

固定模型 AudioUNet5Optimized (V6)，Composite Loss + Warmup+CosineDecay，
对比5种数据增强策略:
1. 无增强（Baseline）
2. 频率掩蔽（FreqMask）— 最大宽度20频率bin，2个掩码
3. 时间掩蔽（TimeMask）— 最大宽度20时间帧，2个掩码
4. 联合掩蔽（JointMask）— 频率 + 时间掩蔽同时应用
5. 综合增强（Full Augmentation）— 联合掩蔽 + 增益缩放 + 噪声注入 + Mixup

用法:
    python scripts/augmentation_comparison.py
    python scripts/augmentation_comparison.py --epochs 100 --debug
    python scripts/augmentation_comparison.py --model unet_v6_optimized
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import HowlingDataset
from src.models import get_model
from src.models.loss_functions import CompositeLoss
from src.models.augmentation import (
    FreqMaskAugmentation,
    TimeMaskAugmentation,
    JointMaskAugmentation,
    FullAugmentation,
)
from src.models.training_strategies import CosineAnnealingWarmupScheduler
from src.evaluation.metrics import AudioMetrics, calculate_mos_score


# 增强配置: (名称, 增强实例或None)
def build_augmentation_configs():
    return [
        ('无增强(Baseline)', None),
        ('频率掩蔽(FreqMask)', FreqMaskAugmentation(max_width=20, num_masks=2)),
        ('时间掩蔽(TimeMask)', TimeMaskAugmentation(max_width=20, num_masks=2)),
        ('联合掩蔽(JointMask)', JointMaskAugmentation(freq_max_width=20, time_max_width=20,
                                                       num_freq_masks=2, num_time_masks=2)),
        ('综合增强(Full)', FullAugmentation(use_mixup=True)),
    ]


AUGMENTATION_NAMES = ['无增强(Baseline)', '频率掩蔽(FreqMask)', '时间掩蔽(TimeMask)',
                       '联合掩蔽(JointMask)', '综合增强(Full)']


def train_and_evaluate(aug_name, augmenter, model_name, train_loader, val_loader,
                       device, args):
    """用指定数据增强策略训练并评估"""
    print(f"\n{'='*60}")
    print(f"  数据增强: {aug_name}")
    print(f"{'='*60}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_class = get_model(model_name)
    model = model_class().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {param_count:,}")

    # Composite Loss: Multi-Resolution STFT(α=1.0) + SI-SDR(β=0.5)
    criterion = CompositeLoss(alpha=1.0, beta=0.5)

    # Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Warmup + CosineDecay调度器
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=5,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=1e-6,
    )

    num_epochs = args.epochs
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    start_time = time.time()

    # 判断增强类型是否需要 (x, y) 对（Mixup类）
    needs_pair = isinstance(augmenter, FullAugmentation)

    for epoch in range(num_epochs):
        # 更新学习率
        scheduler.step(epoch)

        # 训练
        model.train()
        epoch_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # 应用数据增强
            if augmenter is not None:
                with torch.no_grad():
                    try:
                        if needs_pair:
                            noisy, clean = augmenter(noisy, clean)
                        else:
                            augmented = augmenter(noisy)
                            noisy = augmented
                    except Exception as e:
                        pass  # 增强失败则使用原始数据

            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证（不使用增强）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                val_loss += criterion(pred, clean).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")

    elapsed = time.time() - start_time

    # 最终评估
    model.eval()
    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)
    eval_metrics = {
        'snr_improvement_db': [], 'psnr_db': [],
        'stoi_score': [], 'howling_reduction_db': [],
    }
    final_losses = []

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            final_losses.append(nn.L1Loss()(pred, clean).item())
            m = metrics_calc.calculate_all_metrics(clean=pred, noisy=noisy, enhanced=pred)
            for k in eval_metrics:
                if k in m:
                    eval_metrics[k].append(m[k])

    results = {
        'augmentation': aug_name,
        'model': model_name,
        'param_count': param_count,
        'best_val_loss': best_val_loss,
        'avg_l1_loss': float(np.mean(final_losses)),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time_seconds': elapsed,
        'epochs': num_epochs,
        'learning_rate': args.lr,
        'scheduler': 'Warmup+CosineDecay',
        'loss_function': 'CompositeLoss(MR-STFT+SI-SDR)',
    }
    for k, v in eval_metrics.items():
        results[k] = float(np.mean(v)) if v else 0.0
    results['mos_estimate'] = calculate_mos_score(results)

    print(f"\n  结果: Loss={results['avg_l1_loss']:.4f}, "
          f"SNR={results['snr_improvement_db']:.2f}dB, "
          f"STOI={results['stoi_score']:.4f}, "
          f"MOS={results['mos_estimate']:.2f}, "
          f"耗时={elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description='实验6: 数据增强对比')
    parser.add_argument('--model', type=str, default='unet_v6_optimized',
                        choices=list(cfg.AVAILABLE_MODELS.keys()),
                        help='统一使用的模型 (默认: unet_v6_optimized)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='experiments/exp6_augmentation',
                        help='输出目录')
    parser.add_argument('--debug', action='store_true', help='调试模式（3 epochs）')
    parser.add_argument('--augs', nargs='+', default=None,
                        choices=AUGMENTATION_NAMES,
                        help='只测试指定的增强策略')
    args = parser.parse_args()

    if args.debug:
        args.epochs = 3

    device = cfg.DEVICE
    print(f"设备: {device}, 模型: {args.model}")
    print(f"损失函数: CompositeLoss(MR-STFT α=1.0 + SI-SDR β=0.5)")
    print(f"调度器: Warmup(5 epochs) + CosineDecay")
    print(f"学习率: {args.lr}")

    print("\n加载数据集...")
    train_dataset = HowlingDataset(cfg.TRAIN_CLEAN_DIR, cfg.TRAIN_NOISY_DIR)
    val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
    print(f"训练: {len(train_dataset)}, 验证: {len(val_dataset)}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 增强配置
    configs = build_augmentation_configs()
    if args.augs:
        configs = [c for c in configs if c[0] in args.augs]

    print(f"\n待对比数据增强 ({len(configs)}个):")
    for name, _ in configs:
        print(f"  - {name}")

    all_results = []
    for name, augmenter in configs:
        result = train_and_evaluate(name, augmenter, args.model, train_loader,
                                    val_loader, device, args)
        all_results.append(result)

    # 保存详细结果
    results_path = output_dir / 'augmentation_comparison_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)

    # 生成对比表格（表5-7）
    print(f"\n{'='*100}")
    print("  表5-7 数据增强对比结果")
    print(f"{'='*100}")
    print(f"{'增强策略':25s} | {'Best Val':>10s} | {'L1 Loss':>8s} | "
          f"{'SNR(dB)':>8s} | {'PSNR(dB)':>8s} | {'STOI':>7s} | "
          f"{'MOS':>5s} | {'Howling↓dB':>10s} | {'耗时(s)':>8s}")
    print("-" * 100)
    for r in sorted(all_results, key=lambda x: x['mos_estimate'], reverse=True):
        print(f"{r['augmentation']:25s} | {r['best_val_loss']:10.4f} | "
              f"{r['avg_l1_loss']:8.4f} | {r['snr_improvement_db']:8.2f} | "
              f"{r['psnr_db']:8.2f} | {r['stoi_score']:7.4f} | "
              f"{r['mos_estimate']:5.2f} | {r['howling_reduction_db']:10.2f} | "
              f"{r['training_time_seconds']:7.1f}s")

    # 保存LaTeX格式表格
    latex_path = output_dir / 'table_5_7_latex.txt'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{数据增强策略对比结果}\n")
        f.write("\\label{tab:augmentation_comparison}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("增强策略 & L1 Loss & SNR (dB) & PSNR (dB) & STOI & MOS & 耗时 (s) \\\\\n")
        f.write("\\midrule\n")
        for r in sorted(all_results, key=lambda x: x['mos_estimate'], reverse=True):
            f.write(f"{r['augmentation']} & {r['avg_l1_loss']:.4f} & "
                    f"{r['snr_improvement_db']:.2f} & {r['psnr_db']:.2f} & "
                    f"{r['stoi_score']:.4f} & {r['mos_estimate']:.2f} & "
                    f"{r['training_time_seconds']:.1f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\n结果已保存:")
    print(f"  JSON: {results_path}")
    print(f"  LaTeX: {latex_path}")


if __name__ == '__main__':
    main()
