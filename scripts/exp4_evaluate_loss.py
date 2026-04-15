"""实验4: 损失函数对比 — 评估与报告生成

加载 experiments/exp4_loss/ 下训练好的模型，进行全面评估（SNR/STOI/PESQ/SI-SDR/MOS），
生成对比报告和可视化图表，保存到 experiments/exp4_result/。

前置条件:
    先运行训练脚本: python scripts/exp4_train_loss.py

用法:
    python scripts/exp4_evaluate_loss.py
    python scripts/exp4_evaluate_loss.py --train-dir experiments/exp4_loss
    python scripts/exp4_evaluate_loss.py --batch-size 32
    CUDA_VISIBLE_DEVICES=0 python scripts/exp4_evaluate_loss.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import HowlingDataset
from src.evaluate import load_model_from_checkpoint, evaluate_with_metrics

# 固定模型（与训练脚本一致）
FIXED_MODEL = 'unet_v6_optimized'

# 损失函数配置（与训练脚本一致，用于显示名称映射）
LOSS_CONFIGS = [
    ('MSE Loss',                   'mse'),
    ('L1 Loss',                    'l1'),
    ('Spectral Convergence Loss',  'spectral_convergence'),
    ('Multi-Resolution STFT Loss', 'multi_resolution_stft'),
    ('Composite Loss',             'composite'),
]
LOSS_NAME_MAP = {ltype: lname for lname, ltype in LOSS_CONFIGS}

# 默认路径
DEFAULT_TRAIN_DIR = 'experiments/exp4_loss'
DEFAULT_OUTPUT_DIR = 'experiments/exp4_result'


def evaluate_all_models(train_dir: Path, output_dir: Path, batch_size: int, device: torch.device):
    """加载所有训练好的模型并评估"""
    ckpt_dir = train_dir / 'checkpoints'
    if not ckpt_dir.exists():
        print(f"错误: 找不到检查点目录 {ckpt_dir}")
        print(f"请先运行训练脚本: python scripts/exp4_train_loss.py")
        return []

    # 加载训练结果（含训练曲线）
    training_results_path = train_dir / 'training_results.json'
    training_results = {}
    if training_results_path.exists():
        with open(training_results_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        training_results = {r['loss_type']: r for r in training_data}

    # 发现所有检查点
    ckpt_files = sorted(ckpt_dir.glob('best_*.pth'))
    if not ckpt_files:
        print(f"错误: 在 {ckpt_dir} 中未找到检查点文件")
        return []

    print(f"发现 {len(ckpt_files)} 个模型检查点")

    # 准备验证数据集（带波形，用于时域评估）
    print("加载验证数据集（带波形）...")
    val_dataset_td = HowlingDataset(
        cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR,
        return_waveform=True, preload_to_memory=True,
    )
    val_loader_td = DataLoader(
        val_dataset_td, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    all_results = []

    for ckpt_path in ckpt_files:
        # 从文件名提取 loss_type: best_{loss_type}.pth
        loss_type = ckpt_path.stem.replace('best_', '')
        loss_name = LOSS_NAME_MAP.get(loss_type, loss_type)

        print(f"\n{'='*60}")
        print(f"  评估: {loss_name} ({loss_type})")
        print(f"{'='*60}")

        # 加载模型
        model, model_name, ckpt_info = load_model_from_checkpoint(
            str(ckpt_path), FIXED_MODEL, str(device),
        )
        print(f"  模型: {model.__class__.__name__}")

        # 综合评估（含时域指标 STOI/PESQ/SI-SDR）
        print("  计算综合评估指标...")
        metrics = evaluate_with_metrics(model, val_loader_td, str(device), use_timedomain=True)

        # 合并训练信息
        result = {
            'loss_name': loss_name,
            'loss_type': loss_type,
            'model': model_name,
        }

        # 加入训练曲线信息
        if loss_type in training_results:
            tr = training_results[loss_type]
            result['param_count'] = tr.get('param_count', 0)
            result['best_val_loss'] = tr.get('best_val_loss', 0)
            result['train_losses'] = tr.get('train_losses', [])
            result['val_losses'] = tr.get('val_losses', [])
            result['training_time_seconds'] = tr.get('training_time_seconds', 0)
            result['num_epochs'] = tr.get('num_epochs', 0)
        else:
            # 从 checkpoint 中读取
            ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
            result['param_count'] = ckpt.get('param_count', 0)
            result['best_val_loss'] = ckpt.get('best_val_loss', 0)
            result['train_losses'] = ckpt.get('train_losses', [])
            result['val_losses'] = ckpt.get('val_losses', [])
            result['training_time_seconds'] = ckpt.get('training_time_seconds', 0)
            result['num_epochs'] = ckpt.get('num_epochs', 0)

        # 加入评估指标
        result.update(metrics)

        print(f"  结果: SNR={metrics.get('snr_improvement_db', 0):.2f}dB, "
              f"STOI={metrics.get('stoi_score', 0):.4f}, "
              f"PESQ={metrics.get('pesq_score', 0):.2f}, "
              f"MOS={metrics.get('mos_estimate', 0):.2f}")

        all_results.append(result)

        # 释放显存
        del model
        torch.cuda.empty_cache()

    return all_results


def generate_report(all_results, output_dir):
    """生成实验报告（表5-5格式）"""
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("实验4: 损失函数对比结果")
    report_lines.append("固定模型: AudioUNet5Optimized (V6)")
    report_lines.append("=" * 100)

    header = (f"{'损失函数':28s} | {'Best Val':>10s} | {'Avg Loss':>8s} | "
              f"{'SNR(dB)':>8s} | {'PSNR(dB)':>8s} | {'STOI':>7s} | "
              f"{'PESQ':>6s} | {'MOS':>5s} | {'耗时(s)':>8s}")
    report_lines.append(header)
    report_lines.append("-" * 100)

    for r in sorted(all_results, key=lambda x: x.get('mos_estimate', 0), reverse=True):
        row = (f"{r['loss_name']:28s} | {r.get('best_val_loss', 0):10.4f} | "
               f"{r.get('avg_loss', 0):8.4f} | {r.get('snr_improvement_db', 0):8.2f} | "
               f"{r.get('psnr_db', 0):8.2f} | {r.get('stoi_score', 0):7.4f} | "
               f"{r.get('pesq_score', 0):6.2f} | {r.get('mos_estimate', 0):5.2f} | "
               f"{r.get('training_time_seconds', 0):8.1f}")
        report_lines.append(row)

    report_lines.append("-" * 100)

    best = max(all_results, key=lambda x: x.get('mos_estimate', 0))
    report_lines.append(f"\n最佳损失函数: {best['loss_name']} (MOS={best.get('mos_estimate', 0):.2f})")

    report_text = "\n".join(report_lines)
    print(f"\n{report_text}")

    report_path = output_dir / 'report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"报告已保存: {report_path}")

    return report_text


def generate_markdown_table(all_results, output_dir):
    """生成Markdown格式的表格（表5-5）"""
    lines = []
    lines.append("## 表5-5 损失函数对比结果\n")
    lines.append("固定模型: AudioUNet5Optimized (V6)\n")
    lines.append("| 损失函数 | Best Val Loss | Avg Loss | SNR(dB) | PSNR(dB) | STOI | PESQ | MOS | 耗时(s) |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in sorted(all_results, key=lambda x: x.get('mos_estimate', 0), reverse=True):
        lines.append(
            f"| {r['loss_name']} | {r.get('best_val_loss', 0):.4f} | "
            f"{r.get('avg_loss', 0):.4f} | "
            f"{r.get('snr_improvement_db', 0):.2f} | {r.get('psnr_db', 0):.2f} | "
            f"{r.get('stoi_score', 0):.4f} | {r.get('pesq_score', 0):.2f} | "
            f"{r.get('mos_estimate', 0):.2f} | {r.get('training_time_seconds', 0):.1f} |"
        )

    md_text = "\n".join(lines)
    md_path = output_dir / 'table5_5_loss_comparison.md'
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
            train_losses = r.get('train_losses', [])
            if train_losses:
                epochs = list(range(1, len(train_losses) + 1))
                ax1.plot(epochs, train_losses, label=r['loss_name'], linewidth=1.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 验证损失曲线
        ax2 = axes[1]
        for r in all_results:
            val_losses = r.get('val_losses', [])
            if val_losses:
                epochs = list(range(1, len(val_losses) + 1))
                ax2.plot(epochs, val_losses, label=r['loss_name'], linewidth=1.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = output_dir / 'training_curves.png'
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
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4']

        for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
            values = [r.get(metric_key, 0) for r in all_results]
            bars = ax.bar(range(len(names)), values, color=colors[:len(names)],
                          edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            ax.grid(True, alpha=0.3, axis='y')

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        fig_path = output_dir / 'loss_comparison_bar.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"对比柱状图已保存: {fig_path}")
    except ImportError:
        print("matplotlib 不可用，跳过绘图")


def main():
    parser = argparse.ArgumentParser(description='实验4: 损失函数对比 — 评估与报告生成')
    parser.add_argument('--train-dir', type=str, default=DEFAULT_TRAIN_DIR,
                        help='训练输出目录（含 checkpoints/ 和 training_results.json）')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='评估结果输出目录')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='评估批大小')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    train_dir = PROJECT_ROOT / args.train_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 评估所有模型
    all_results = evaluate_all_models(train_dir, output_dir, args.batch_size, device)
    if not all_results:
        print("未找到可评估的模型，退出。")
        return

    # 2. 保存完整评估结果
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)
    print(f"\n评估结果已保存: {results_path}")

    # 3. 生成报告
    generate_report(all_results, output_dir)
    generate_markdown_table(all_results, output_dir)

    # 4. 生成可视化
    plot_training_curves(all_results, output_dir)
    plot_comparison_bar(all_results, output_dir)

    print(f"\n所有结果已保存至: {output_dir}")


if __name__ == '__main__':
    main()