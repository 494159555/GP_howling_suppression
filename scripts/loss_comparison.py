"""实验4: 损失函数对比

固定模型为 AudioUNet5Optimized (V6)，仅替换损失函数，对比训练效果:
1. MSE Loss          — 均方误差，基础回归损失
2. L1 Loss           — 平均绝对误差，对异常值不敏感
3. SI-SDR Loss       — 尺度不变信噪比负值，信号级损失
4. Multi-Resolution STFT Loss — 多分辨率(帧长512/256/128)频谱L1损失
5. Composite Loss    — 多分辨率STFT(α=1.0) + SI-SDR(β=0.5)

用法:
    python scripts/loss_comparison.py
    python scripts/loss_comparison.py --epochs 100
    python scripts/loss_comparison.py --debug
    python scripts/loss_comparison.py --losses MSE L1
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
from src.models.loss_functions import (
    SISDRLoss, MultiResolutionSTFTLoss, CompositeLoss,
)
from src.evaluation.metrics import AudioMetrics, calculate_mos_score


# 实验4要求的5种损失函数配置
LOSS_CONFIGS = [
    ('MSE Loss',                   'mse'),
    ('L1 Loss',                    'l1'),
    ('SI-SDR Loss',                'si_sdr'),
    ('Multi-Resolution STFT Loss', 'multi_resolution_stft'),
    ('Composite Loss',             'composite'),
]

# 固定模型
FIXED_MODEL = 'unet_v6_optimized'


def get_criterion(loss_type: str):
    """获取损失函数实例"""
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'si_sdr':
        return SISDRLoss()
    elif loss_type == 'multi_resolution_stft':
        return MultiResolutionSTFTLoss()
    elif loss_type == 'composite':
        return CompositeLoss(alpha=1.0, beta=0.5)
    else:
        raise ValueError(f"未知损失函数: {loss_type}")


def train_one_loss(loss_name, loss_type, model_name, train_loader, val_loader,
                   device, args):
    """用指定损失函数训练并评估模型"""
    print(f"\n{'='*70}")
    print(f"  训练损失函数: {loss_name} ({loss_type})")
    print(f"  固定模型: {model_name}")
    print(f"{'='*70}")

    # 每次创建新模型，保证公平对比（固定随机种子）
    model_class = get_model(model_name)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = model_class().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {param_count:,}")

    criterion = get_criterion(loss_type)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_state = None
    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(args.epochs):
        # --- 训练 ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            if isinstance(loss, tuple):
                loss = loss[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                loss = criterion(pred, clean)
                if isinstance(loss, tuple):
                    loss = loss[0]
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    Epoch [{epoch+1:3d}/{args.epochs}] "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_now:.2e}")

    elapsed = time.time() - start_time

    # --- 加载最佳模型进行统一评估 ---
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)
    eval_metrics = {
        'snr_improvement_db': [],
        'psnr_db': [],
        'stoi_score': [],
        'howling_reduction_db': [],
    }
    final_l1_losses = []
    final_mse_losses = []

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            final_l1_losses.append(nn.L1Loss()(pred, clean).item())
            final_mse_losses.append(nn.MSELoss()(pred, clean).item())
            m = metrics_calc.calculate_all_metrics(clean=clean, noisy=noisy, enhanced=pred)
            for k in eval_metrics:
                if k in m:
                    eval_metrics[k].append(m[k])

    results = {
        'loss_name': loss_name,
        'loss_type': loss_type,
        'model': model_name,
        'param_count': param_count,
        'best_val_loss': float(best_val_loss),
        'avg_l1_loss': float(np.mean(final_l1_losses)),
        'avg_mse_loss': float(np.mean(final_mse_losses)),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'training_time_seconds': float(elapsed),
        'num_epochs': args.epochs,
    }
    for k, v in eval_metrics.items():
        results[k] = float(np.mean(v)) if v else 0.0
    results['mos_estimate'] = float(calculate_mos_score(results))

    print(f"\n  结果: L1={results['avg_l1_loss']:.4f}, "
          f"SNR={results['snr_improvement_db']:.2f}dB, "
          f"STOI={results['stoi_score']:.4f}, "
          f"MOS={results['mos_estimate']:.2f}, "
          f"耗时={elapsed:.1f}s")

    # 保存最佳模型检查点
    ckpt_dir = Path(args.output_dir) / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'best_{loss_type}.pth'
    torch.save({
        'model_state_dict': best_state,
        'loss_type': loss_type,
        'loss_name': loss_name,
        'best_val_loss': best_val_loss,
        'results': {k: v for k, v in results.items()
                    if k not in ('train_losses', 'val_losses')},
    }, ckpt_path)

    return results


def generate_report(all_results, output_dir):
    """生成实验报告（表5-5格式）"""
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("实验4: 损失函数对比结果")
    report_lines.append(f"固定模型: AudioUNet5Optimized (V6)")
    report_lines.append("=" * 100)

    # 表5-5格式
    header = (f"{'损失函数':28s} | {'Best Val':>10s} | {'L1 Loss':>8s} | "
              f"{'SNR(dB)':>8s} | {'PSNR(dB)':>8s} | {'STOI':>7s} | "
              f"{'MOS':>5s} | {'耗时(s)':>8s}")
    report_lines.append(header)
    report_lines.append("-" * 100)

    for r in sorted(all_results, key=lambda x: x['mos_estimate'], reverse=True):
        row = (f"{r['loss_name']:28s} | {r['best_val_loss']:10.4f} | "
               f"{r['avg_l1_loss']:8.4f} | {r['snr_improvement_db']:8.2f} | "
               f"{r['psnr_db']:8.2f} | {r['stoi_score']:7.4f} | "
               f"{r['mos_estimate']:5.2f} | {r['training_time_seconds']:8.1f}")
        report_lines.append(row)

    report_lines.append("-" * 100)

    # 找出最佳损失函数
    best = max(all_results, key=lambda x: x['mos_estimate'])
    report_lines.append(f"\n最佳损失函数: {best['loss_name']} (MOS={best['mos_estimate']:.2f})")

    report_text = "\n".join(report_lines)
    print(f"\n{report_text}")

    # 保存报告
    report_path = Path(output_dir) / 'report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    return report_text


def generate_markdown_table(all_results, output_dir):
    """生成Markdown格式的表格（表5-5）"""
    lines = []
    lines.append("## 表5-5 损失函数对比结果\n")
    lines.append(f"固定模型: AudioUNet5Optimized (V6)\n")
    lines.append("| 损失函数 | Best Val Loss | L1 Loss | SNR(dB) | PSNR(dB) | STOI | MOS | 耗时(s) |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in sorted(all_results, key=lambda x: x['mos_estimate'], reverse=True):
        lines.append(
            f"| {r['loss_name']} | {r['best_val_loss']:.4f} | {r['avg_l1_loss']:.4f} | "
            f"{r['snr_improvement_db']:.2f} | {r['psnr_db']:.2f} | {r['stoi_score']:.4f} | "
            f"{r['mos_estimate']:.2f} | {r['training_time_seconds']:.1f} |"
        )

    md_text = "\n".join(lines)
    md_path = Path(output_dir) / 'table5_5_loss_comparison.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f"Markdown表格已保存: {md_path}")


def plot_training_curves(all_results, output_dir):
    """绘制训练曲线"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 训练损失曲线
        ax1 = axes[0]
        for r in all_results:
            epochs = list(range(1, len(r['train_losses']) + 1))
            ax1.plot(epochs, r['train_losses'], label=r['loss_name'], linewidth=1.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 验证损失曲线
        ax2 = axes[1]
        for r in all_results:
            epochs = list(range(1, len(r['val_losses']) + 1))
            ax2.plot(epochs, r['val_losses'], label=r['loss_name'], linewidth=1.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = Path(output_dir) / 'training_curves.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存: {fig_path}")
    except ImportError:
        print("matplotlib 不可用，跳过绘图")


def plot_comparison_bar(all_results, output_dir):
    """绘制损失函数对比柱状图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        names = [r['loss_name'] for r in all_results]
        metrics_to_plot = [
            ('snr_improvement_db', 'SNR Improvement (dB)'),
            ('stoi_score', 'STOI Score'),
            ('mos_estimate', 'MOS Estimate'),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

        for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
            values = [r[metric_key] for r in all_results]
            bars = ax.bar(range(len(names)), values, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            ax.grid(True, alpha=0.3, axis='y')

            # 在柱子上标注数值
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        fig_path = Path(output_dir) / 'loss_comparison_bar.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"对比柱状图已保存: {fig_path}")
    except ImportError:
        print("matplotlib 不可用，跳过绘图")


def main():
    parser = argparse.ArgumentParser(description='实验4: 损失函数对比')
    parser.add_argument('--model', type=str, default=FIXED_MODEL,
                        help=f'模型 (固定: {FIXED_MODEL})')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='experiments/exp4_loss_comparison',
                        help='输出目录')
    parser.add_argument('--debug', action='store_true', help='调试模式（3 epochs）')
    parser.add_argument('--losses', nargs='+', default=None,
                        choices=[c[1] for c in LOSS_CONFIGS],
                        help='只测试指定的损失函数类型')
    args = parser.parse_args()

    if args.debug:
        args.epochs = 3
        args.batch_size = 4

    device = cfg.DEVICE
    print(f"设备: {device}")
    print(f"模型: {args.model}")
    print(f"训练轮数: {args.epochs}, 批大小: {args.batch_size}, 学习率: {args.lr}")

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = HowlingDataset(cfg.TRAIN_CLEAN_DIR, cfg.TRAIN_NOISY_DIR)
    val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 筛选要测试的损失函数
    configs = LOSS_CONFIGS
    if args.losses:
        configs = [c for c in configs if c[1] in args.losses]

    print(f"\n待对比损失函数 ({len(configs)}个):")
    for name, ltype in configs:
        print(f"  - {name} ({ltype})")

    # 逐个训练评估
    all_results = []
    for name, ltype in configs:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        result = train_one_loss(
            name, ltype, args.model,
            train_loader, val_loader, device, args
        )
        all_results.append(result)

    # 保存原始结果JSON
    results_path = output_dir / 'loss_comparison_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)
    print(f"\n结果JSON已保存: {results_path}")

    # 生成报告
    generate_report(all_results, output_dir)
    generate_markdown_table(all_results, output_dir)

    # 生成可视化
    plot_training_curves(all_results, output_dir)
    plot_comparison_bar(all_results, output_dir)

    print(f"\n所有结果已保存至: {output_dir}")


if __name__ == '__main__':
    main()
