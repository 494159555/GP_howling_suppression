"""实验4: 损失函数对比

在统一模型（AudioUNet5）上对比不同损失函数的训练效果:
- L1 Loss
- MSE Loss
- Spectral Loss（频谱损失）
- MultiTask Loss（多任务: spectral+L1+MSE）
- MultiTask Consistency Loss（多任务+频谱一致性）

用法:
    python scripts/loss_comparison.py
    python scripts/loss_comparison.py --epochs 50 --debug
    python scripts/loss_comparison.py --model unet_v6_optimized
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
from src.models.loss_functions import SpectralLoss, MultiTaskLoss
from src.evaluation.metrics import AudioMetrics, calculate_mos_score
from src.train import get_loss_function


# 损失函数配置
LOSS_CONFIGS = [
    ('L1',                  'l1'),
    ('MSE',                 'mse'),
    ('Spectral',            'spectral'),
    ('MultiTask',           'multitask'),
    ('MultiTask+Consist',   'multitask_consistency'),
]


def get_criterion(loss_type):
    """获取损失函数实例"""
    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'spectral':
        return SpectralLoss()
    elif loss_type == 'multitask':
        return _WrapMultiTask(MultiTaskLoss(
            weights={'spectral': 0.5, 'l1': 0.3, 'mse': 0.2},
            use_consistency=False,
        ))
    elif loss_type == 'multitask_consistency':
        return _WrapMultiTask(MultiTaskLoss(
            weights={'spectral': 0.5, 'l1': 0.3, 'mse': 0.2},
            use_consistency=True,
        ))
    else:
        raise ValueError(f"未知损失函数: {loss_type}")


class _WrapMultiTask(nn.Module):
    """包装多任务损失，只返回总loss"""
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, pred, target):
        result = self.base(pred, target)
        if isinstance(result, tuple):
            return result[0]
        return result


def train_and_evaluate(loss_name, loss_type, model_name, train_loader, val_loader,
                       device, args):
    """用指定损失函数训练并评估模型"""
    print(f"\n{'='*60}")
    print(f"  损失函数: {loss_name} ({loss_type})")
    print(f"{'='*60}")

    # 每次创建新模型，保证公平对比
    model_class = get_model(model_name)
    torch.manual_seed(args.seed)
    model = model_class().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型: {model.__class__.__name__}, 参数量: {param_count:,}")

    criterion = get_criterion(loss_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    num_epochs = args.epochs
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(num_epochs):
        # 训练
        model.train()
        epoch_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(noisy)
            loss = criterion(pred, clean)
            if isinstance(loss, tuple):
                loss = loss[0]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                loss = criterion(pred, clean)
                if isinstance(loss, tuple):
                    loss = loss[0]
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    elapsed = time.time() - start_time

    # 最终评估（用L1 loss做统一评估）
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
        'loss_name': loss_name,
        'loss_type': loss_type,
        'model': model_name,
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

    print(f"\n  结果: L1={results['avg_l1_loss']:.4f}, "
          f"SNR={results['snr_improvement_db']:.2f}dB, "
          f"MOS={results['mos_estimate']:.2f}, "
          f"耗时={elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description='实验4: 损失函数对比')
    parser.add_argument('--model', type=str, default='unet_v2',
                        choices=list(cfg.AVAILABLE_MODELS.keys()),
                        help='统一使用的模型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='experiments/exp4_loss_comparison',
                        help='输出目录')
    parser.add_argument('--debug', action='store_true', help='调试模式（3 epochs）')
    parser.add_argument('--losses', nargs='+', default=None,
                        choices=['L1', 'MSE', 'Spectral', 'MultiTask', 'MultiTask+Consist'],
                        help='只测试指定的损失函数')
    args = parser.parse_args()

    if args.debug:
        args.epochs = 3

    device = cfg.DEVICE
    print(f"设备: {device}, 模型: {args.model}")

    # 数据
    print("\n加载数据集...")
    train_dataset = HowlingDataset(cfg.TRAIN_CLEAN_DIR, cfg.TRAIN_NOISY_DIR)
    val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"训练: {len(train_dataset)}, 验证: {len(val_dataset)}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 筛选损失函数
    configs = LOSS_CONFIGS
    if args.losses:
        configs = [c for c in configs if c[0] in args.losses]

    print(f"\n待对比损失函数 ({len(configs)}个):")
    for name, _ in configs:
        print(f"  - {name}")

    # 运行实验
    all_results = []
    for name, ltype in configs:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        result = train_and_evaluate(name, ltype, args.model, train_loader, val_loader, device, args)
        all_results.append(result)

    # 保存结果
    results_path = output_dir / 'loss_comparison_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)

    # 打印对比表格
    print(f"\n{'='*85}")
    print("  损失函数对比结果")
    print(f"{'='*85}")
    print(f"{'损失函数':20s} | {'Best Val':>10s} | {'L1 Loss':>8s} | {'SNR':>8s} | {'STOI':>7s} | {'MOS':>5s} | {'耗时':>8s}")
    print("-" * 85)
    for r in sorted(all_results, key=lambda x: x['mos_estimate'], reverse=True):
        print(f"{r['loss_name']:20s} | {r['best_val_loss']:10.4f} | "
              f"{r['avg_l1_loss']:8.4f} | {r['snr_improvement_db']:8.2f} | "
              f"{r['stoi_score']:7.4f} | {r['mos_estimate']:5.2f} | "
              f"{r['training_time_seconds']:7.1f}s")

    print(f"\n结果已保存: {results_path}")


if __name__ == '__main__':
    main()
