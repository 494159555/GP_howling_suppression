"""实验4: 损失函数对比 — 训练阶段

固定模型为 AudioUNet5Optimized (V6)，仅替换损失函数，逐个训练并保存模型。
评估与报告生成请使用 scripts/exp4_evaluate_loss.py。

损失函数列表:
1. MSE Loss          — 均方误差，基础回归损失
2. L1 Loss           — 平均绝对误差，对异常值不敏感（等价于对数域L1/频谱损失）
3. Spectral Convergence Loss — 频谱收敛损失，相对Frobenius范数误差
4. Multi-Resolution STFT Loss — 多分辨率(帧长512/256/128)频谱L1损失
5. Composite Loss    — 多分辨率STFT(α=1.0) + 频谱收敛(β=0.5)

用法:
    python scripts/exp4_train_loss.py
    python scripts/exp4_train_loss.py --epochs 80
    python scripts/exp4_train_loss.py --debug
    python scripts/exp4_train_loss.py --losses MSE L1
    CUDA_VISIBLE_DEVICES=0,1 python scripts/exp4_train_loss.py --gpus 2
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import HowlingDataset
from src.models import get_model
from src.models.loss_functions import (
    SpectralConvergenceLoss, MultiResolutionSTFTLoss, CompositeLoss,
)
# 复用 src/train.py 中的调度器工具函数
from src.train import get_scheduler

# 实验4要求的损失函数配置
LOSS_CONFIGS = [
    ('MSE Loss',                   'mse'),
    ('L1 Loss',                    'l1'),
    ('Spectral Convergence Loss',  'spectral_convergence'),
    ('Multi-Resolution STFT Loss', 'multi_resolution_stft'),
    ('Composite Loss',             'composite'),
]

# 固定模型
FIXED_MODEL = 'unet_v6_optimized'

# 默认输出目录
DEFAULT_OUTPUT_DIR = 'experiments/exp4_loss'


def get_criterion(loss_type: str):
    """获取损失函数实例"""
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'spectral_convergence':
        return SpectralConvergenceLoss()
    elif loss_type == 'multi_resolution_stft':
        return MultiResolutionSTFTLoss()
    elif loss_type == 'composite':
        return CompositeLoss(alpha=1.0, beta=0.5)
    else:
        raise ValueError(f"未知损失函数: {loss_type}")


def train_one_loss(loss_name, loss_type, model_name, train_loader, val_loader,
                   device, args, n_gpus=1):
    """用指定损失函数训练模型，保存最佳checkpoint和训练曲线"""
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

    # 多GPU DataParallel
    if n_gpus > 1 and torch.cuda.device_count() >= n_gpus:
        model = nn.DataParallel(model)
        print(f"  使用 DataParallel: {n_gpus} GPUs")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {param_count:,}")

    criterion = get_criterion(loss_type)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = get_scheduler(
        optimizer, 'plateau', args.epochs,
    )

    # AMP混合精度训练 - 利用Tensor Core加速，减少显存占用
    scaler = GradScaler()

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
            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = model(noisy)
                loss = criterion(pred, clean)
                if isinstance(loss, tuple):
                    loss = loss[0]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
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
                noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
                with autocast():
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
            raw_state = model.state_dict()
            best_state = {k.replace('module.', ''): v.cpu().clone() for k, v in raw_state.items()}

        lr_now = optimizer.param_groups[0]['lr']
        print(f"    Epoch [{epoch+1:3d}/{args.epochs}] "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_now:.2e}")

    elapsed = time.time() - start_time

    # 保存最佳模型检查点
    ckpt_dir = Path(args.output_dir) / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'best_{loss_type}.pth'
    torch.save({
        'model_state_dict': best_state,
        'loss_type': loss_type,
        'loss_name': loss_name,
        'model_name': model_name,
        'best_val_loss': best_val_loss,
        'param_count': param_count,
        'num_epochs': args.epochs,
        'training_time_seconds': elapsed,
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
    }, ckpt_path)

    print(f"\n  训练完成: Best Val Loss={best_val_loss:.4f}, 耗时={elapsed:.1f}s")
    print(f"  模型已保存: {ckpt_path}")

    # 释放GPU显存
    del model, best_state, criterion, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'loss_name': loss_name,
        'loss_type': loss_type,
        'model': model_name,
        'param_count': param_count,
        'best_val_loss': float(best_val_loss),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'training_time_seconds': float(elapsed),
        'num_epochs': args.epochs,
    }


def main():
    parser = argparse.ArgumentParser(description='实验4: 损失函数对比 — 训练阶段')
    parser.add_argument('--model', type=str, default=FIXED_MODEL,
                        help=f'模型 (固定: {FIXED_MODEL})')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='每卡批大小')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='输出目录')
    parser.add_argument('--debug', action='store_true', help='调试模式（3 epochs）')
    parser.add_argument('--gpus', type=int, default=1, help='使用的GPU数量')
    parser.add_argument('--losses', nargs='+', default=None,
                        choices=[c[1] for c in LOSS_CONFIGS],
                        help='只训练指定的损失函数类型')
    args = parser.parse_args()

    if args.debug:
        args.epochs = 3
        args.batch_size = 4
        args.gpus = min(args.gpus, 2)

    # 多GPU配置
    n_gpus = args.gpus
    effective_batch_size = args.batch_size * n_gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 启用TF32加速（Ampere架构，19位浮点，~2-3x矩阵运算加速，精度损失可忽略）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 启用cuDNN自动调优（选择最快的卷积算法）
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print(f"设备: {device}")
    print(f"GPU数量: {n_gpus}, 每卡batch_size: {args.batch_size}, "
          f"有效batch_size: {effective_batch_size}")
    print(f"模型: {args.model}")
    print(f"训练轮数: {args.epochs}, 学习率: {args.lr}")

    # 加载数据集（预加载到内存，消除磁盘I/O瓶颈）
    print("\n加载数据集（预加载到内存）...")
    train_dataset = HowlingDataset(cfg.TRAIN_CLEAN_DIR, cfg.TRAIN_NOISY_DIR, preload_to_memory=True)
    val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR, preload_to_memory=True)
    # 数据已预加载到内存，num_workers=0 + pin_memory避免IPC开销
    loader_kwargs = dict(num_workers=0, pin_memory=True)
    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=effective_batch_size, shuffle=False,
        **loader_kwargs,
    )
    print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 筛选要测试的损失函数
    configs = LOSS_CONFIGS
    if args.losses:
        configs = [c for c in configs if c[1] in args.losses]

    print(f"\n待训练损失函数 ({len(configs)}个):")
    for name, ltype in configs:
        print(f"  - {name} ({ltype})")

    # 逐个训练（每完成一个就保存，支持断点续跑）
    results_path = output_dir / 'training_results.json'
    all_results = []
    total_count = len(configs)

    # 如果已有中间结果，加载续跑
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        completed_types = {r['loss_type'] for r in all_results}
        configs = [(n, t) for n, t in configs if t not in completed_types]
        total_count = len(all_results) + len(configs)
        print(f"  已加载 {len(all_results)} 个已完成的结果，剩余 {len(configs)} 个待训练（共 {total_count} 个）")

    for name, ltype in configs:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        result = train_one_loss(
            name, ltype, args.model,
            train_loader, val_loader, device, args,
            n_gpus=n_gpus,
        )
        all_results.append(result)

        # 增量保存：每完成一个损失函数就写入JSON
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)
        print(f"  已保存中间结果 ({len(all_results)}/{total_count}): {results_path}")

    print(f"\n所有训练完成！结果保存在: {output_dir}")
    print(f"下一步请运行评估脚本: python scripts/exp4_evaluate_loss.py")


if __name__ == '__main__':
    main()