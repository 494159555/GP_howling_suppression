"""实验7: 数据增强对比（对应EXPERIMENTS.md阶段七）

固定设置:
  - 模型: AudioUNet5Optimized (unet_v6_optimized)
  - 损失函数: Composite Loss (MultiTaskLoss: spectral=0.5, l1=0.3, mse=0.2)
  - 学习率调度: Warmup + CosineDecay (5 epochs warmup, lr 1e-3 -> 1e-6)

对比5种增强策略:
  1. 无增强（Baseline）
  2. 频率掩蔽（FreqMask）- 最大宽度20频率bin，2个掩码
  3. 时间掩蔽（TimeMask）- 最大宽度20时间帧，2个掩码
  4. 联合掩蔽（JointMask）- 频率 + 时间掩蔽同时应用
  5. 综合增强（Full Augmentation）- 联合掩蔽 + 增益缩放(0.8~1.2) + 噪声注入(SNR 20~40dB) + Mixup(α=0.4)

用法:
    python scripts/augmentation_comparison.py
    python scripts/augmentation_comparison.py --epochs 50 --debug
    python scripts/augmentation_comparison.py --gpu 0
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
from src.models.loss_functions import MultiTaskLoss
from src.models.augmentation import SpecAugment, MixupAugmentation
from src.models.training_strategies import CosineAnnealingWarmupScheduler
from src.evaluation.metrics import AudioMetrics, calculate_mos_score


# ============ 增强策略定义 ============

class FreqMaskOnly:
    """仅频率掩蔽 - 最大宽度20频率bin，2个掩码"""

    def __init__(self):
        self.aug = SpecAugment(
            freq_mask_param=20, time_mask_param=0,
            num_freq_masks=2, num_time_masks=0, prob=1.0
        )

    def __call__(self, x):
        return self.aug(x)


class TimeMaskOnly:
    """仅时间掩蔽 - 最大宽度20时间帧，2个掩码"""

    def __init__(self):
        self.aug = SpecAugment(
            freq_mask_param=0, time_mask_param=20,
            num_freq_masks=0, num_time_masks=2, prob=1.0
        )

    def __call__(self, x):
        return self.aug(x)


class JointMask:
    """联合掩蔽 - 频率 + 时间掩蔽同时应用"""

    def __init__(self):
        self.aug = SpecAugment(
            freq_mask_param=20, time_mask_param=20,
            num_freq_masks=2, num_time_masks=2, prob=1.0
        )

    def __call__(self, x):
        return self.aug(x)


class FullAugmentation:
    """综合增强 - 联合掩蔽 + 增益缩放(0.8~1.2) + 噪声注入(SNR 20~40dB) + Mixup(α=0.4)"""

    def __init__(self):
        self.mask = JointMask()
        self.mixup = MixupAugmentation(alpha=0.4, prob=0.5)

    def _gain_scaling(self, x):
        """增益缩放 0.8~1.2"""
        import random
        factor = random.uniform(0.8, 1.2)
        return x * factor

    def _noise_injection(self, x, snr_db_range=(20, 40)):
        """噪声注入 SNR 20~40dB"""
        import random
        snr_db = random.uniform(*snr_db_range)
        signal_power = torch.mean(x ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise

    def __call__(self, x, target=None, x2=None, target2=None):
        """应用综合增强

        Args:
            x: 输入频谱
            target: 目标频谱（可选）
            x2: 第二个样本输入（Mixup用）
            target2: 第二个样本目标（Mixup用）
        """
        # 1. 联合掩蔽
        x = self.mask(x)
        # 2. 增益缩放
        x = self._gain_scaling(x)
        # 3. 噪声注入
        x = self._noise_injection(x)
        # 4. Mixup（如果有第二个样本）
        if x2 is not None and target is not None and target2 is not None:
            x, target, _, _ = self.mixup(x, target, x2, target2)
            return x, target
        return x


def build_augmentation_configs():
    """构建5种增强策略配置"""
    return [
        ('None(基线)', None),
        ('FreqMask', FreqMaskOnly()),
        ('TimeMask', TimeMaskOnly()),
        ('JointMask', JointMask()),
        ('FullAugmentation', FullAugmentation()),
    ]


AUGMENTATION_NAMES = ['None(基线)', 'FreqMask', 'TimeMask', 'JointMask', 'FullAugmentation']


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

    # Composite Loss (MultiTaskLoss)
    criterion = MultiTaskLoss(weights={'spectral': 0.5, 'l1': 0.3, 'mse': 0.2})
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Warmup + CosineDecay 学习率调度器
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        warmup_epochs=5,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=1e-6
    )

    num_epochs = args.epochs
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    start_time = time.time()

    # 用于Mixup的额外数据迭代器
    use_full_aug = (aug_name == 'FullAugmentation')

    for epoch in range(num_epochs):
        # 训练
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # 如果是FullAugmentation，需要两个dataloader同步迭代
        if use_full_aug:
            train_iter = iter(train_loader)
            train_iter2 = iter(train_loader)

            for i, (noisy, clean) in enumerate(train_loader):
                noisy, clean = noisy.to(device), clean.to(device)

                # 应用综合增强（含Mixup）
                try:
                    # 获取第二组样本
                    try:
                        noisy2, clean2 = next(train_iter2)
                    except StopIteration:
                        train_iter2 = iter(train_loader)
                        noisy2, clean2 = next(train_iter2)
                    noisy2, clean2 = noisy2.to(device), clean2.to(device)

                    result = augmenter(noisy, clean, noisy2, clean2)
                    if isinstance(result, tuple) and len(result) == 2:
                        noisy, clean = result[0], result[1]
                    else:
                        noisy = result
                except Exception:
                    pass

                optimizer.zero_grad()
                pred = model(noisy)
                loss, _ = criterion(pred, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
        else:
            for noisy, clean in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)

                # 应用数据增强
                if augmenter is not None:
                    with torch.no_grad():
                        try:
                            augmented = augmenter(noisy)
                            if isinstance(augmented, tuple):
                                noisy = augmented[0]
                            else:
                                noisy = augmented
                        except Exception:
                            pass

                optimizer.zero_grad()
                pred = model(noisy)
                loss, _ = criterion(pred, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

        train_loss = epoch_loss / num_batches
        train_losses.append(train_loss)

        # 验证（不使用增强）
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                vloss, _ = criterion(pred, clean)
                val_loss += vloss.item()
                val_batches += 1
        val_loss /= val_batches
        val_losses.append(val_loss)

        # 更新学习率
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")

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
        'loss_function': 'CompositeLoss(spectral=0.5,l1=0.3,mse=0.2)',
        'lr_scheduler': 'WarmupCosineDecay(warmup=5,lr=1e-3->1e-6)',
        'param_count': param_count,
        'best_val_loss': best_val_loss,
        'avg_l1_loss': float(np.mean(final_losses)),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time_seconds': elapsed,
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
    parser = argparse.ArgumentParser(description='实验7: 数据增强对比')
    parser.add_argument('--model', type=str, default='unet_v6_optimized',
                        choices=list(cfg.AVAILABLE_MODELS.keys()),
                        help='统一使用的模型（默认: unet_v6_optimized）')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')
    parser.add_argument('--output-dir', type=str, default='experiments/exp7_augmentation',
                        help='输出目录')
    parser.add_argument('--debug', action='store_true', help='调试模式（3 epochs）')
    parser.add_argument('--augs', nargs='+', default=None,
                        choices=AUGMENTATION_NAMES,
                        help='只测试指定的增强策略')
    args = parser.parse_args()

    if args.debug:
        args.epochs = 3

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}, 模型: {args.model}")
    print(f"损失函数: CompositeLoss (spectral=0.5, l1=0.3, mse=0.2)")
    print(f"学习率调度: Warmup+CosineDecay (warmup=5, lr={args.lr}->1e-6)")

    print("\n加载数据集...")
    train_dataset = HowlingDataset(cfg.TRAIN_CLEAN_DIR, cfg.TRAIN_NOISY_DIR)
    val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
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
        result = train_and_evaluate(name, augmenter, args.model, train_loader, val_loader, device, args)
        all_results.append(result)

    # 保存详细结果
    results_path = output_dir / 'augmentation_comparison_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)

    # 生成对比表格
    print(f"\n{'='*100}")
    print("  表5-7: 数据增强对比结果")
    print(f"  模型: AudioUNet5Optimized (V6) | 损失: Composite Loss | 调度: Warmup+CosineDecay")
    print(f"{'='*100}")
    print(f"{'增强策略':20s} | {'Best Val':>10s} | {'L1 Loss':>8s} | {'SNR(dB)':>8s} | "
          f"{'STOI':>7s} | {'Howling(dB)':>12s} | {'MOS':>5s} | {'耗时':>8s}")
    print("-" * 100)
    for r in sorted(all_results, key=lambda x: x['mos_estimate'], reverse=True):
        print(f"{r['augmentation']:20s} | {r['best_val_loss']:10.4f} | "
              f"{r['avg_l1_loss']:8.4f} | {r['snr_improvement_db']:8.2f} | "
              f"{r['stoi_score']:7.4f} | {r['howling_reduction_db']:12.2f} | "
              f"{r['mos_estimate']:5.2f} | "
              f"{r['training_time_seconds']:7.1f}s")

    # 保存Markdown格式表格
    md_path = output_dir / 'table_5_7_augmentation_comparison.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 表5-7: 数据增强对比结果\n\n")
        f.write(f"**模型**: AudioUNet5Optimized (V6)  \n")
        f.write(f"**损失函数**: Composite Loss (spectral=0.5, l1=0.3, mse=0.2)  \n")
        f.write(f"**学习率调度**: Warmup + CosineDecay (warmup=5 epochs, lr=1e-3 → 1e-6)  \n\n")
        f.write(f"| 增强策略 | Best Val Loss | L1 Loss | SNR改善(dB) | STOI | 啸叫抑制(dB) | MOS | 训练耗时(s) |\n")
        f.write(f"|:---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(all_results, key=lambda x: x['mos_estimate'], reverse=True):
            f.write(f"| {r['augmentation']} | {r['best_val_loss']:.4f} | "
                    f"{r['avg_l1_loss']:.4f} | {r['snr_improvement_db']:.2f} | "
                    f"{r['stoi_score']:.4f} | {r['howling_reduction_db']:.2f} | "
                    f"{r['mos_estimate']:.2f} | {r['training_time_seconds']:.1f} |\n")
    print(f"\n结果已保存:")
    print(f"  JSON: {results_path}")
    print(f"  表格: {md_path}")


if __name__ == '__main__':
    main()
