"""实验5: 训练策略对比（严格匹配EXPERIMENTS.md规范）

固定模型 AudioUNet5Optimized (V6)，损失函数 Composite Loss，
对比5种学习率调度策略:
  1. CosineAnnealingLR: lr 1e-3 → 1e-6 余弦衰减
  2. ReduceLROnPlateau: patience=5, factor=0.5, min_lr=1e-6
  3. CyclicLR: lr 1e-5 ~ 1e-3 三角循环
  4. OneCycleLR: 前30%升至1e-3，后70%退火至1e-6
  5. Warmup + CosineDecay: 前5轮线性预热至1e-3，后余弦衰减至1e-6

统一训练参数:
  batch_size=16, lr=1e-3, epochs=100, early_stopping=15
  Adam, AMP混合精度, 梯度裁剪=1.0, weight_decay=1e-5

用法:
    python scripts/exp5_training_strategy.py
    python scripts/exp5_training_strategy.py --epochs 50 --debug
    python scripts/exp5_training_strategy.py --strategies CosineAnnealing Plateau
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import HowlingDataset
from src.models import get_model
from src.models.loss_functions import CompositeLoss
from src.evaluation.metrics import AudioMetrics, calculate_mos_score


# ==================== 学习率调度策略 ====================

class CosineAnnealingLR:
    """纯余弦退火，无warmup

    lr: base_lr → min_lr，余弦衰减
    """

    def __init__(self, optimizer, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        progress = self.current_epoch / self.total_epochs
        new_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class WarmupCosineDecay:
    """Warmup + 余弦衰减

    前 warmup_epochs 轮线性预热至 base_lr，后余弦衰减至 min_lr
    """

    def __init__(self, optimizer, total_epochs, base_lr, warmup_epochs=5, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # 线性预热
            scale = self.current_epoch / self.warmup_epochs
            new_lr = self.base_lr * scale
        else:
            # 余弦衰减
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            new_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ==================== 策略定义 ====================

STRATEGY_CONFIGS = [
    ('CosineAnnealing', {
        'description': 'CosineAnnealingLR: lr 1e-3 → 1e-6 余弦衰减',
    }),
    ('Plateau', {
        'description': 'ReduceLROnPlateau: patience=5, factor=0.5, min_lr=1e-6',
    }),
    ('CyclicLR', {
        'description': 'CyclicLR: lr 1e-5 ~ 1e-3 三角循环',
    }),
    ('OneCycle', {
        'description': 'OneCycleLR: 前30%升至1e-3，后70%退火至1e-6',
    }),
    ('WarmupCosine', {
        'description': 'Warmup + CosineDecay: 前5轮线性预热至1e-3，后余弦衰减至1e-6',
    }),
]


def build_scheduler(name, optimizer, total_epochs, base_lr=1e-3, steps_per_epoch=None):
    """根据策略名称构建学习率调度器"""
    if name == 'CosineAnnealing':
        return CosineAnnealingLR(optimizer, total_epochs, base_lr, min_lr=1e-6)

    elif name == 'Plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

    elif name == 'CyclicLR':
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=1e-5, max_lr=1e-3,
            step_size_up=(steps_per_epoch or 100) * 5, mode='triangular',
            cycle_momentum=False
        )

    elif name == 'OneCycle':
        total_steps = (steps_per_epoch or 100) * total_epochs
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=total_steps,
            pct_start=0.3, anneal_strategy='cos',
            div_factor=25.0, final_div_factor=1e4
        )

    elif name == 'WarmupCosine':
        return WarmupCosineDecay(
            optimizer, total_epochs, base_lr, warmup_epochs=5, min_lr=1e-6
        )

    else:
        raise ValueError(f"未知策略: {name}")


def train_one_strategy(strategy_name, model_name, train_loader, val_loader, val_loader_td, device, args):
    """训练并评估单个策略"""
    desc = dict(STRATEGY_CONFIGS)[strategy_name]
    print(f"\n{'='*70}")
    print(f"  策略: {strategy_name}")
    print(f"  {desc}")
    print(f"{'='*70}")

    # 固定种子，保证公平对比
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建模型
    model_class = get_model(model_name)
    model = model_class().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型: {model.__class__.__name__}, 参数量: {param_count:,}")

    # 损失函数: CompositeLoss (Multi-Resolution STFT + SI-SDR)
    criterion = CompositeLoss(alpha=1.0, beta=0.5)

    # 优化器: Adam, weight_decay=1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 学习率调度器（先用placeholder，在第一个epoch确定steps_per_epoch后重建）
    scheduler = build_scheduler(strategy_name, optimizer, args.epochs, base_lr=args.lr, steps_per_epoch=len(train_loader))

    # AMP
    scaler = GradScaler()
    use_amp = args.amp

    # 早停
    early_stopping = EarlyStopping(patience=args.patience)

    # 记录
    train_losses = []
    val_losses = []
    lr_history = []
    best_val_loss = float('inf')
    best_model_state = None
    start_time = time.time()

    for epoch in range(args.epochs):
        # ---- 训练 ----
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    pred = model(noisy)
                    loss = criterion(pred, clean)
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(noisy)
                loss = criterion(pred, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # CyclicLR / OneCycleLR 需要batch-level step
            if strategy_name in ('CyclicLR', 'OneCycle'):
                scheduler.step()

        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                loss = criterion(pred, clean)
                val_loss += loss.item()
                n_val += 1

        val_loss /= max(n_val, 1)
        val_losses.append(val_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # 调度器 step (epoch-level)
        if strategy_name == 'Plateau':
            scheduler.step(val_loss)
        elif strategy_name in ('CosineAnnealing', 'WarmupCosine'):
            scheduler.step()
        # CyclicLR 和 OneCycleLR 已在 batch level step

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 日志
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1:3d}/{args.epochs}] "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")

        # 早停
        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            print(f"    早停于 epoch {epoch+1}")
            break

    elapsed = time.time() - start_time

    # ---- 最终评估（加载最佳模型） ----
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    model.eval()
    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)
    eval_metrics = {
        'snr_improvement_db': [],
        'psnr_db': [],
        'stoi_score': [],
        'si_sdr_db': [],
        'pesq_score': [],
        'howling_reduction_db': [],
    }
    final_l1_losses = []

    # 优先使用时域评估（STOI/PESQ需要时域信号）
    use_td = val_loader_td is not None
    with torch.no_grad():
        td_iter = iter(val_loader_td) if use_td else None
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            final_l1_losses.append(nn.L1Loss()(pred, clean).item())

            if use_td:
                try:
                    td_batch = next(td_iter)
                    td_noisy, td_clean, td_noisy_wave, td_clean_wave, td_noisy_stft = td_batch
                    td_noisy = td_noisy.to(device)
                    td_pred = model(td_noisy)
                    from src.evaluate import _istft_from_mag_phase
                    enhanced_wave = _istft_from_mag_phase(td_pred.cpu(), td_noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                    noisy_wave_td = _istft_from_mag_phase(td_noisy.cpu(), td_noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                    # 时域信号质量指标
                    m = {
                        'snr_improvement_db': metrics_calc.calculate_snr(td_clean_wave, enhanced_wave, noisy_wave_td),
                        'psnr_db': metrics_calc.calculate_psnr(td_clean_wave, enhanced_wave),
                        'si_sdr_db': metrics_calc.calculate_si_sdr(td_clean_wave, enhanced_wave),
                        'stoi_score': metrics_calc.calculate_stoi(td_clean_wave, enhanced_wave),
                        'pesq_score': metrics_calc.calculate_pesq(td_clean_wave, enhanced_wave),
                    }
                    # 啸叫抑制指标需要频谱数据
                    howling_m = metrics_calc.calculate_howling_reduction(td_noisy.cpu(), td_pred.cpu())
                    m.update(howling_m)
                except Exception:
                    m = metrics_calc.calculate_all_metrics(clean=clean, noisy=noisy, enhanced=pred)
            else:
                m = metrics_calc.calculate_all_metrics(clean=clean, noisy=noisy, enhanced=pred)

            for k in eval_metrics:
                if k in m:
                    eval_metrics[k].append(m[k])

    results = {
        'strategy_name': strategy_name,
        'description': desc,
        'model': model_name,
        'param_count': param_count,
        'best_val_loss': best_val_loss,
        'avg_l1_loss': float(np.mean(final_l1_losses)),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lr_history': lr_history,
        'training_time_seconds': round(elapsed, 1),
        'actual_epochs': len(train_losses),
    }
    for k, v in eval_metrics.items():
        results[k] = float(np.mean(v)) if v else 0.0
    results['mos_estimate'] = calculate_mos_score(results)

    print(f"\n  结果: Best Val={best_val_loss:.4f}, L1={results['avg_l1_loss']:.4f}, "
          f"SNR={results['snr_improvement_db']:.2f}dB, "
          f"MOS={results['mos_estimate']:.2f}, "
          f"耗时={elapsed:.1f}s ({len(train_losses)} epochs)")

    # 保存最佳模型
    model_save_dir = Path(args.output_dir) / 'checkpoints'
    model_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model_state, model_save_dir / f'{strategy_name}_best.pt')

    return results


def generate_comparison_table(results):
    """生成对比表格"""
    print(f"\n{'='*110}")
    print("  实验5: 训练策略对比结果")
    print(f"  模型: AudioUNet5Optimized (V6) | 损失: CompositeLoss (MR-STFT + SI-SDR)")
    print(f"{'='*110}")
    header = (f"{'策略':18s} | {'Epochs':>6s} | {'Best Val':>10s} | {'L1 Loss':>8s} | "
              f"{'SNR(dB)':>8s} | {'STOI':>7s} | {'MOS':>5s} | {'耗时(s)':>8s}")
    print(header)
    print("-" * 110)

    for r in sorted(results, key=lambda x: x['mos_estimate'], reverse=True):
        row = (f"{r['strategy_name']:18s} | {r['actual_epochs']:6d} | "
               f"{r['best_val_loss']:10.4f} | {r['avg_l1_loss']:8.4f} | "
               f"{r['snr_improvement_db']:8.2f} | {r['stoi_score']:7.4f} | "
               f"{r['mos_estimate']:5.2f} | {r['training_time_seconds']:8.1f}")
        print(row)
    print("=" * 110)


def generate_markdown_report(results, output_dir):
    """生成Markdown格式报告"""
    report_path = output_dir / 'experiment5_report.md'

    # 按MOS排序
    sorted_results = sorted(results, key=lambda x: x['mos_estimate'], reverse=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 实验5: 训练策略对比\n\n")
        f.write("## 实验设置\n\n")
        f.write("- **模型**: AudioUNet5Optimized (V6)\n")
        f.write("- **损失函数**: Composite Loss (Multi-Resolution STFT α=1.0 + SI-SDR β=0.5)\n")
        f.write("- **优化器**: Adam, weight_decay=1e-5\n")
        f.write("- **初始学习率**: 1e-3\n")
        f.write("- **批大小**: 16\n")
        f.write("- **最大轮数**: 100\n")
        f.write("- **早停耐心**: 15\n")
        f.write("- **混合精度**: 开启\n")
        f.write("- **梯度裁剪**: 1.0\n\n")

        f.write("## 对比策略\n\n")
        f.write("| 序号 | 策略 | 说明 |\n")
        f.write("|:---:|:---|:---|\n")
        f.write("| 1 | CosineAnnealingLR | lr: 1e-3 → 1e-6，余弦衰减 |\n")
        f.write("| 2 | ReduceLROnPlateau | patience=5, factor=0.5, min_lr=1e-6 |\n")
        f.write("| 3 | CyclicLR | lr: 1e-5 ~ 1e-3，三角循环 |\n")
        f.write("| 4 | OneCycleLR | 前30%升至1e-3，后70%退火至1e-6 |\n")
        f.write("| 5 | Warmup + CosineDecay | 前5轮线性预热至1e-3，后余弦衰减至1e-6 |\n\n")

        f.write("## 结果对比（表5-6）\n\n")
        f.write("| 策略 | Epochs | Best Val Loss | L1 Loss | SNR(dB) | STOI | MOS | 耗时(s) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in sorted_results:
            f.write(f"| {r['strategy_name']} | {r['actual_epochs']} | "
                    f"{r['best_val_loss']:.4f} | {r['avg_l1_loss']:.4f} | "
                    f"{r['snr_improvement_db']:.2f} | {r['stoi_score']:.4f} | "
                    f"{r['mos_estimate']:.2f} | {r['training_time_seconds']:.1f} |\n")

        # 最佳策略
        best = sorted_results[0]
        f.write(f"\n## 结论\n\n")
        f.write(f"**最佳策略**: {best['strategy_name']} (MOS={best['mos_estimate']:.2f})\n\n")
        f.write(f"- {best['strategy_name']} 在验证集上取得最佳MOS评分 {best['mos_estimate']:.2f}\n")
        f.write(f"- SNR提升: {best['snr_improvement_db']:.2f} dB\n")
        f.write(f"- STOI: {best['stoi_score']:.4f}\n")
        f.write(f"- 训练耗时: {best['training_time_seconds']:.1f} 秒 ({best['actual_epochs']} epochs)\n\n")

        # 各策略详细分析
        f.write("## 各策略训练曲线\n\n")
        for r in sorted_results:
            f.write(f"### {r['strategy_name']}\n\n")
            f.write(f"- 最终训练Loss: {r['train_losses'][-1]:.4f}\n")
            f.write(f"- 最终验证Loss: {r['val_losses'][-1]:.4f}\n")
            f.write(f"- 初始LR: {r['lr_history'][0]:.2e}\n")
            f.write(f"- 最终LR: {r['lr_history'][-1]:.2e}\n\n")

    print(f"  Markdown报告: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='实验5: 训练策略对比')
    parser.add_argument('--model', type=str, default='unet_v6_optimized',
                        choices=list(cfg.AVAILABLE_MODELS.keys()),
                        help='统一使用的模型（默认: unet_v6_optimized）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数（默认: 100）')
    parser.add_argument('--batch-size', type=int, default=16, help='批大小（默认: 16）')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率（默认: 1e-3）')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心（默认: 15）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='experiments/exp5_training_strategy',
                        help='输出目录')
    parser.add_argument('--debug', action='store_true', help='调试模式（5 epochs）')
    parser.add_argument('--no-amp', action='store_true', help='禁用AMP')
    parser.add_argument('--strategies', nargs='+', default=None,
                        choices=[s[0] for s in STRATEGY_CONFIGS],
                        help='只测试指定策略')
    args = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.patience = 3

    args.amp = not args.no_amp
    device = cfg.DEVICE

    print("=" * 70)
    print("  实验5: 训练策略对比")
    print("=" * 70)
    print(f"  设备: {device}")
    print(f"  模型: {args.model}")
    print(f"  损失: CompositeLoss (MR-STFT + SI-SDR)")
    print(f"  初始LR: {args.lr}, Batch: {args.batch_size}, Epochs: {args.epochs}")
    print(f"  AMP: {args.amp}, 早停耐心: {args.patience}")

    # 数据
    print("\n加载数据集...")
    train_dataset = HowlingDataset(cfg.TRAIN_CLEAN_DIR, cfg.TRAIN_NOISY_DIR)
    val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
    # 带波形的验证集用于时域评估（STOI/PESQ/SI-SDR）
    val_dataset_td = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR, return_waveform=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    val_loader_td = DataLoader(val_dataset_td, batch_size=args.batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)
    print(f"  训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    # 输出目录
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 筛选策略
    configs = STRATEGY_CONFIGS
    if args.strategies:
        configs = [c for c in configs if c[0] in args.strategies]

    print(f"\n待对比策略 ({len(configs)}个):")
    for name, params in configs:
        print(f"  - {name}: {params['description']}")

    # 运行所有策略
    all_results = []
    for name, params in configs:
        result = train_one_strategy(
            name, args.model, train_loader, val_loader, val_loader_td, device, args
        )
        all_results.append(result)

    # 保存原始结果
    results_path = output_dir / 'strategy_comparison_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)

    # 生成对比表格
    generate_comparison_table(all_results)

    # 生成Markdown报告
    generate_markdown_report(all_results, output_dir)

    # 保存实验配置
    config_path = output_dir / 'experiment_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'exp5_training_strategy_comparison',
            'model': args.model,
            'loss': 'CompositeLoss (MR-STFT alpha=1.0 + SI-SDR beta=0.5)',
            'optimizer': 'Adam',
            'weight_decay': 1e-5,
            'initial_lr': args.lr,
            'batch_size': args.batch_size,
            'max_epochs': args.epochs,
            'patience': args.patience,
            'amp': args.amp,
            'gradient_clip': 1.0,
            'seed': args.seed,
            'strategies': [c[0] for c in configs],
        }, f, indent=4, ensure_ascii=False)

    print(f"\n所有结果已保存至: {output_dir}")
    print(f"  - 原始数据: {results_path}")
    print(f"  - 实验配置: {config_path}")
    print(f"  - 模型权重: {output_dir / 'checkpoints'}/")


if __name__ == '__main__':
    main()
