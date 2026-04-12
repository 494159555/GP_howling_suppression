"""实验5: 训练策略对比

在统一模型和损失函数下对比不同训练策略:
- LR调度器: plateau / cosine_warmup / one_cycle / step
- 混合精度训练: 开启 / 关闭
- Warmup轮数: 0 / 3 / 5 / 10
- 课程学习: 开启 / 关闭

用法:
    python scripts/training_strategy_comparison.py
    python scripts/training_strategy_comparison.py --epochs 50 --debug
    python scripts/training_strategy_comparison.py --model unet_v6_optimized
"""

import argparse
import json
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
from src.models.loss_functions import SpectralLoss
from src.evaluation.metrics import AudioMetrics, calculate_mos_score
from src.train import get_scheduler


# 训练策略配置: (名称, 参数字典)
STRATEGY_CONFIGS = [
    # 学习率调度器对比
    ('LR_Plateau',        {'lr_scheduler': 'plateau', 'mixed_precision': False, 'warmup_epochs': 0}),
    ('LR_CosineWarmup',   {'lr_scheduler': 'cosine_warmup', 'mixed_precision': False, 'warmup_epochs': 5}),
    ('LR_OneCycle',       {'lr_scheduler': 'one_cycle', 'mixed_precision': False, 'warmup_epochs': 0}),
    ('LR_Step',           {'lr_scheduler': 'step', 'mixed_precision': False, 'warmup_epochs': 0}),
    # 混合精度
    ('AMP_On',            {'lr_scheduler': 'plateau', 'mixed_precision': True, 'warmup_epochs': 0}),
    # Warmup对比
    ('Warmup_3',          {'lr_scheduler': 'cosine_warmup', 'mixed_precision': False, 'warmup_epochs': 3}),
    ('Warmup_10',         {'lr_scheduler': 'cosine_warmup', 'mixed_precision': False, 'warmup_epochs': 10}),
    # 组合策略
    ('CosWarmup+AMP',     {'lr_scheduler': 'cosine_warmup', 'mixed_precision': True, 'warmup_epochs': 5}),
    ('OneCycle+AMP',      {'lr_scheduler': 'one_cycle', 'mixed_precision': True, 'warmup_epochs': 0}),
]


def train_and_evaluate(strategy_name, strategy_params, model_name,
                       train_loader, val_loader, device, args):
    """用指定训练策略训练并评估"""
    print(f"\n{'='*60}")
    print(f"  训练策略: {strategy_name}")
    print(f"  参数: {strategy_params}")
    print(f"{'='*60}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_class = get_model(model_name)
    model = model_class().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {param_count:,}")

    criterion = SpectralLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    use_amp = strategy_params.get('mixed_precision', False)
    scaler = GradScaler() if use_amp else None

    lr_scheduler_type = strategy_params.get('lr_scheduler', 'plateau')
    warmup_epochs = strategy_params.get('warmup_epochs', 0)
    num_epochs = args.epochs

    scheduler = get_scheduler(
        optimizer, lr_scheduler_type, num_epochs,
        warmup_epochs=warmup_epochs,
    )

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

            if use_amp and scaler is not None:
                with autocast():
                    pred = model(noisy)
                    loss = criterion(pred, clean)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(noisy)
                loss = criterion(pred, clean)
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
                val_loss += criterion(pred, clean).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if lr_scheduler_type == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}")

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
        'strategy_name': strategy_name,
        'strategy_params': strategy_params,
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

    print(f"\n  结果: Loss={results['avg_l1_loss']:.4f}, "
          f"SNR={results['snr_improvement_db']:.2f}dB, "
          f"MOS={results['mos_estimate']:.2f}, "
          f"耗时={elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description='实验5: 训练策略对比')
    parser.add_argument('--model', type=str, default='unet_v2',
                        choices=list(cfg.AVAILABLE_MODELS.keys()),
                        help='统一使用的模型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='experiments/exp5_strategy_comparison',
                        help='输出目录')
    parser.add_argument('--debug', action='store_true', help='调试模式（3 epochs）')
    parser.add_argument('--strategies', nargs='+', default=None,
                        choices=[s[0] for s in STRATEGY_CONFIGS],
                        help='只测试指定的策略')
    args = parser.parse_args()

    if args.debug:
        args.epochs = 3

    device = cfg.DEVICE
    print(f"设备: {device}, 模型: {args.model}")

    print("\n加载数据集...")
    train_dataset = HowlingDataset(cfg.TRAIN_CLEAN_DIR, cfg.TRAIN_NOISY_DIR)
    val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"训练: {len(train_dataset)}, 验证: {len(val_dataset)}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 筛选策略
    configs = STRATEGY_CONFIGS
    if args.strategies:
        configs = [c for c in configs if c[0] in args.strategies]

    print(f"\n待对比训练策略 ({len(configs)}个):")
    for name, params in configs:
        print(f"  - {name}: {params}")

    all_results = []
    for name, params in configs:
        result = train_and_evaluate(name, params, args.model, train_loader, val_loader, device, args)
        all_results.append(result)

    # 保存
    results_path = output_dir / 'strategy_comparison_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)

    # 对比表格
    print(f"\n{'='*95}")
    print("  训练策略对比结果")
    print(f"{'='*95}")
    print(f"{'策略':20s} | {'Best Val':>10s} | {'L1 Loss':>8s} | {'SNR':>8s} | {'MOS':>5s} | {'耗时':>8s}")
    print("-" * 95)
    for r in sorted(all_results, key=lambda x: x['mos_estimate'], reverse=True):
        print(f"{r['strategy_name']:20s} | {r['best_val_loss']:10.4f} | "
              f"{r['avg_l1_loss']:8.4f} | {r['snr_improvement_db']:8.2f} | "
              f"{r['mos_estimate']:5.2f} | {r['training_time_seconds']:7.1f}s")

    print(f"\n结果已保存: {results_path}")


if __name__ == '__main__':
    main()
