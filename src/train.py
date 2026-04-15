"""训练模块

音频啸叫抑制模型训练脚本，支持实验管理、监控和可视化
支持4种U-Net变体、多种损失函数、训练策略和数据增强
"""

import argparse
from datetime import datetime
import os
import shutil
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml
from torch.cuda.amp import GradScaler, autocast

from src.config import cfg
from src.dataset import HowlingDataset

# 导入所有模型类
from src.models.unet_v1 import AudioUNet3
from src.models.unet_v2 import AudioUNet5
from src.models.unet_v3_attention import AudioUNet5Attention
from src.models.unet_v6_optimized import AudioUNet5Optimized

# 导入损失函数
try:
    from src.models.loss_functions import (
        SpectralLoss, SpectralConsistencyLoss,
        MultiTaskLoss, AdversarialLoss,
        SpectralConvergenceLoss, MultiResolutionSTFTLoss, CompositeLoss
    )
except ImportError as e:
    print(f"⚠️ 警告: 无法导入部分损失函数模块 - {e}")
    SpectralLoss = None
    SpectralConsistencyLoss = None
    MultiTaskLoss = None
    AdversarialLoss = None
    SpectralConvergenceLoss = None
    MultiResolutionSTFTLoss = None
    CompositeLoss = None

# 导入训练策略（可选）
try:
    from src.models.training_strategies import (
        CosineAnnealingWarmupScheduler,
        OneCycleScheduler
    )
except ImportError:
    CosineAnnealingWarmupScheduler = None
    OneCycleScheduler = None


def setup_ddp():
    """初始化 DDP 分布式训练环境

    通过 torchrun 启动时，环境变量 MASTER_ADDR / MASTER_PORT /
    WORLD_SIZE / RANK / LOCAL_RANK 已自动设置。

    Returns:
        (rank, local_rank, world_size)
    """
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # 必须在 init_process_group 之前设置设备，否则 NCCL 可能无法正确初始化
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, local_rank, world_size


def cleanup_ddp():
    """清理 DDP 分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = 0) -> bool:
    """判断是否为主进程"""
    return rank == 0


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='音频啸叫抑制模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单卡训练
  python src/train.py                                    # 使用默认配置
  python src/train.py --model unet_v6_optimized          # 训练指定模型

  # DDP 多卡训练 (推荐，使用 torchrun)
  torchrun --nproc_per_node=4 src/train.py --model unet_v2
  torchrun --nproc_per_node=4 src/train.py --config configs/unet_v2.yaml --lr 2e-4

  # 指定 GPU（通过环境变量）
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 src/train.py
        """
    )

    # 模型选择
    parser.add_argument(
        '--model',
        type=str,
        default=cfg.DEFAULT_MODEL,
        choices=list(cfg.AVAILABLE_MODELS.keys()),
        help=f'选择模型 (默认: {cfg.DEFAULT_MODEL})'
    )

    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径 (YAML格式)'
    )

    # 训练参数
    parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                       help='学习率')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='每个GPU的批大小')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='每个GPU的数据加载线程数')

    # 损失函数
    parser.add_argument('--loss', type=str, default=None,
                       choices=list(cfg.LOSS_FUNCTIONS.keys()),
                       help='损失函数类型')

    # 训练策略
    parser.add_argument('--mixed-precision', action='store_true',
                       help='启用混合精度训练')
    parser.add_argument('--no-amp', action='store_true',
                       help='禁用混合精度训练')
    parser.add_argument('--lr-scheduler', type=str, default=None,
                       choices=['plateau', 'cosine_warmup', 'one_cycle', 'step'],
                       help='学习率调度器类型')
    parser.add_argument('--warmup-epochs', type=int, default=None,
                       help='Warmup轮数')
    parser.add_argument('--curriculum', action='store_true',
                       help='启用课程学习')

    # 数据增强
    parser.add_argument('--augment', action='store_true',
                       help='启用数据增强')
    parser.add_argument('--no-augment', action='store_true',
                       help='禁用数据增强')
    parser.add_argument('--spec-augment', action='store_true',
                       help='启用SpecAugment')
    parser.add_argument('--mixup', action='store_true',
                       help='启用Mixup增强')

    # 后处理
    parser.add_argument('--post-process', action='store_true',
                       help='启用后处理')

    # 早停
    parser.add_argument('--early-stop', type=int, default=None,
                       help='早停轮数（0=禁用）')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                       help='早停最小改进阈值')

    # 实验管理
    parser.add_argument('--exp-name', type=str, default=None,
                       help='实验名称 (默认自动生成)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')

    # 多GPU训练（DataParallel 模式，DDP 模式请使用 torchrun 启动）
    parser.add_argument('--gpus', type=str, default=None,
                       help='指定使用的GPU (DataParallel模式)，例如 "0,1,2,3"')

    # 调试
    parser.add_argument('--debug', action='store_true',
                       help='调试模式（快速运行）')
    parser.add_argument('--profile', action='store_true',
                       help='性能分析模式')

    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置，支持嵌套结构

    支持的顶层键:
        - model: 模型配置
        - training: 训练参数
        - loss: 损失函数配置
        - training_strategies: 训练策略
        - data_augmentation: 数据增强
        - post_processing: 后处理

    Returns:
        扁平化的配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 扁平化嵌套配置
    flat_config = {}

    if 'model' in config:
        model_cfg = config['model']
        if 'name' in model_cfg:
            flat_config['model'] = model_cfg['name']

    if 'training' in config:
        train_cfg = config['training']
        key_map = {
            'batch_size': 'batch_size',
            'learning_rate': 'learning_rate',
            'lr': 'learning_rate',
            'epochs': 'num_epochs',
            'num_workers': 'num_workers',
        }
        for yaml_key, cfg_key in key_map.items():
            if yaml_key in train_cfg:
                flat_config[cfg_key] = train_cfg[yaml_key]

    if 'loss' in config:
        loss_cfg = config['loss']
        if 'type' in loss_cfg:
            flat_config['loss_function'] = loss_cfg['type']
        if 'weights' in loss_cfg:
            flat_config['loss_weights'] = loss_cfg['weights']

    if 'training_strategies' in config:
        strategy_cfg = config['training_strategies']
        if 'mixed_precision' in strategy_cfg:
            flat_config['mixed_precision'] = strategy_cfg['mixed_precision']
        if 'lr_scheduler' in strategy_cfg:
            flat_config['lr_scheduler'] = strategy_cfg['lr_scheduler']
        if 'warmup_epochs' in strategy_cfg:
            flat_config['warmup_epochs'] = strategy_cfg['warmup_epochs']
        if 'curriculum_learning' in strategy_cfg:
            flat_config['curriculum_learning'] = strategy_cfg['curriculum_learning']

    if 'data_augmentation' in config:
        aug_cfg = config['data_augmentation']
        if 'enabled' in aug_cfg:
            flat_config['data_augmentation'] = aug_cfg['enabled']
        if 'spec_augment' in aug_cfg:
            flat_config['spec_augment'] = aug_cfg['spec_augment']
        if 'mixup' in aug_cfg:
            flat_config['mixup'] = aug_cfg['mixup']

    if 'post_processing' in config:
        post_cfg = config['post_processing']
        if 'enabled' in post_cfg:
            flat_config['post_processing'] = post_cfg['enabled']
        if 'method' in post_cfg:
            flat_config['post_processing_method'] = post_cfg['method']

    return flat_config


def get_model_class(model_name: str) -> type:
    """根据模型名称获取模型类

    Args:
        model_name: 模型名称（如 'unet_v2'）

    Returns:
        模型类
    """
    model_class_name = cfg.AVAILABLE_MODELS.get(model_name)
    if model_class_name is None:
        raise ValueError(f"未知模型: {model_name}")

    # 所有可用的模型类（仅包含实际存在的模型）
    model_classes = [
        AudioUNet3,
        AudioUNet5,
        AudioUNet5Attention,
        AudioUNet5Optimized,
    ]

    for cls in model_classes:
        if cls.__name__ == model_class_name:
            return cls

    raise ValueError(f"未找到模型类: {model_class_name}")


def get_loss_function(loss_type: str, loss_weights: Optional[Dict] = None) -> nn.Module:
    """获取损失函数

    Args:
        loss_type: 损失函数类型
        loss_weights: 损失权重（用于多任务损失）

    Returns:
        损失函数实例
    """

    # 包装类，用于处理返回元组的损失函数
    class LossWrapper(nn.Module):
        def __init__(self, base_loss):
            super().__init__()
            self.base_loss = base_loss

        def forward(self, pred, target):
            result = self.base_loss(pred, target)
            if isinstance(result, tuple):
                return result[0]  # 返回总损失
            return result

    if loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'spectral':
        if SpectralLoss is None:
            print("⚠️ 警告: SpectralLoss 不可用，使用L1损失")
            return nn.L1Loss()
        return SpectralLoss()
    elif loss_type in ['multitask', 'multitask_consistency']:
        if MultiTaskLoss is None:
            print("⚠️ 警告: MultiTaskLoss 不可用，使用L1损失")
            return nn.L1Loss()
        weights = loss_weights or {'spectral': 0.5, 'l1': 0.3, 'mse': 0.2}
        use_consistency = (loss_type == 'multitask_consistency')
        multitask_loss = MultiTaskLoss(weights=weights, use_consistency=use_consistency)
        return LossWrapper(multitask_loss)
    elif loss_type == 'adversarial':
        if AdversarialLoss is None:
            print("⚠️ 警告: AdversarialLoss 不可用，使用L1损失")
            return nn.L1Loss()
        return AdversarialLoss()
    elif loss_type == 'spectral_convergence':
        if SpectralConvergenceLoss is None:
            print("⚠️ 警告: SpectralConvergenceLoss 不可用，使用L1损失")
            return nn.L1Loss()
        return SpectralConvergenceLoss()
    elif loss_type == 'multi_resolution_stft':
        if MultiResolutionSTFTLoss is None:
            print("⚠️ 警告: MultiResolutionSTFTLoss 不可用，使用L1损失")
            return nn.L1Loss()
        return MultiResolutionSTFTLoss()
    elif loss_type == 'composite':
        if CompositeLoss is None:
            print("⚠️ 警告: CompositeLoss 不可用，使用L1损失")
            return nn.L1Loss()
        return CompositeLoss(alpha=1.0, beta=0.5)
    else:
        # 默认使用L1损失
        return nn.L1Loss()


def get_scheduler(optimizer, scheduler_type: str, num_epochs: int, **kwargs):
    """获取学习率调度器

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        num_epochs: 训练轮数
        **kwargs: 额外参数

    Returns:
        学习率调度器
    """
    if scheduler_type == 'cosine_warmup':
        if CosineAnnealingWarmupScheduler is None:
            print("⚠️ 警告: CosineAnnealingWarmupScheduler 不可用，使用默认调度器")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=cfg.SCHEDULER_FACTOR,
                patience=cfg.SCHEDULER_PATIENCE, verbose=True
            )
        warmup_epochs = kwargs.get('warmup_epochs', cfg.WARMUP_EPOCHS)
        min_lr = kwargs.get('min_lr', cfg.COSINE_MIN_LR)
        base_lr = kwargs.get('base_lr', optimizer.param_groups[0]['lr'])
        return CosineAnnealingWarmupScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
            base_lr=base_lr,
            min_lr=min_lr
        )
    elif scheduler_type == 'one_cycle':
        if OneCycleScheduler is None:
            print("⚠️ 警告: OneCycleScheduler 不可用，使用默认调度器")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=cfg.SCHEDULER_FACTOR,
                patience=cfg.SCHEDULER_PATIENCE, verbose=True
            )
        max_lr = kwargs.get('max_lr', cfg.ONE_CYCLE_MAX_LR)
        pct_start = kwargs.get('pct_start', cfg.ONE_CYCLE_PCT_START)
        return OneCycleScheduler(
            optimizer,
            max_lr=max_lr,
            total_epochs=num_epochs,
            pct_start=pct_start
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1
        )
    else:  # plateau (默认)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.SCHEDULER_FACTOR,
            patience=cfg.SCHEDULER_PATIENCE,
            verbose=True
        )


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    scaler=None, use_amp=False):
    """训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        scaler: GradScaler（混合精度）
        use_amp: 是否使用混合精度

    Returns:
        平均损失
    """
    model.train()

    total_loss = 0.0

    for noisy_mag, clean_mag in dataloader:
        noisy_mag = noisy_mag.to(device)
        clean_mag = clean_mag.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast():
                pred_mag = model(noisy_mag)
                loss = criterion(pred_mag, clean_mag)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_mag = model(noisy_mag)
            loss = criterion(pred_mag, clean_mag)

            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证模型

    Args:
        model: 模型
        dataloader: 验证数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        平均损失
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for noisy_mag, clean_mag in dataloader:
            noisy_mag = noisy_mag.to(device)
            clean_mag = clean_mag.to(device)

            pred_mag = model(noisy_mag)
            loss = criterion(pred_mag, clean_mag)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def train() -> None:
    """训练模型（支持 DDP 和 DataParallel 两种多卡模式）

    单卡训练:
        python -m src.train
        python -m src.train --model unet_v2

    DataParallel 多卡训练 (简单，用 --gpus 指定):
        python -m src.train --gpus 0,1,2,3
        python -m src.train --gpus 0,1,2,3 --model unet_v2

    DDP 多卡训练 (推荐，性能更好，用 torchrun 启动):
        torchrun --nproc_per_node=4 -m src.train --model unet_v2
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m src.train
    """
    args = parse_args()

    # ========== 多 GPU 配置 ==========
    # --gpus 参数设置 CUDA_VISIBLE_DEVICES（必须在任何 CUDA 操作之前）
    gpu_ids = None
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        gpu_ids = list(range(len(args.gpus.split(','))))
        # 注意：设置 CUDA_VISIBLE_DEVICES 后，gpu_id 是重新编号的 0,1,2,...

    # ========== DDP 初始化（torchrun 模式） ==========
    ddp_available = dist.is_available() and torch.cuda.is_available()
    use_ddp = ddp_available and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

    if use_ddp:
        rank, local_rank, world_size = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
        is_main = is_main_process(rank)
        if is_main:
            print(f"🖥️ DDP 分布式训练: {world_size} GPUs, rank={rank}, local_rank={local_rank}")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        if gpu_ids is not None and len(gpu_ids) > 1:
            print(f"🖥️ DataParallel 训练: {args.gpus} (共 {len(gpu_ids)} 张卡)")
        elif torch.cuda.is_available():
            print(f"🖥️ 单卡训练: {device}")
        else:
            print(f"🖥️ CPU 训练")

    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # 调试模式：减少轮数和样本数
    if args.debug:
        if is_main:
            print("🐛 调试模式：使用较少的轮数和数据")
        args.epochs = 3
        args.batch_size = 2

    # ========== 性能优化 ==========
    # 启用 cuDNN autotuner（输入尺寸固定时自动选择最快卷积算法）
    torch.backends.cudnn.benchmark = True

    # 1. 加载配置
    config_override = {}

    # YAML配置文件
    if args.config is not None:
        file_config = load_config_from_yaml(args.config)
        config_override.update(file_config)
        if is_main:
            print(f"📄 加载配置文件: {args.config}")

    # 模型名称
    model_name = config_override.get('model', args.model)

    # 命令行参数覆盖
    if args.lr is not None:
        config_override['learning_rate'] = args.lr
    if args.batch_size is not None:
        config_override['batch_size'] = args.batch_size
    if args.epochs is not None:
        config_override['num_epochs'] = args.epochs
    if args.num_workers is not None:
        config_override['num_workers'] = args.num_workers
    if args.loss is not None:
        config_override['loss_function'] = args.loss
    if args.mixed_precision or (not args.no_amp and config_override.get('mixed_precision', False)):
        config_override['mixed_precision'] = True
    if args.lr_scheduler is not None:
        config_override['lr_scheduler'] = args.lr_scheduler
    if args.warmup_epochs is not None:
        config_override['warmup_epochs'] = args.warmup_epochs
    if args.curriculum:
        config_override['curriculum_learning'] = True
    if args.augment or config_override.get('data_augmentation', False):
        config_override['data_augmentation'] = True
    if args.spec_augment:
        config_override['spec_augment'] = True
    if args.mixup:
        config_override['mixup'] = True
    if args.post_process:
        config_override['post_processing'] = True

    # 应用配置（仅主进程打印）
    if config_override and is_main:
        print("\n📝 使用自定义配置:")
        for key, value in sorted(config_override.items()):
            print(f"  {key}: {value}")

    # ========== 2. 实验环境初始化（仅主进程） ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name if args.exp_name else model_name
    experiment_name = f"exp_{timestamp}_{exp_name}"

    exp_dir = cfg.EXP_DIR / experiment_name
    checkpoint_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"

    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 备份配置
        if args.config and os.path.exists(args.config):
            shutil.copy(args.config, exp_dir / "config_backup.yaml")
            print(f"✅ 配置已备份")

        src_config_path = os.path.join("src", "config.py")
        if os.path.exists(src_config_path):
            shutil.copy(src_config_path, exp_dir / "config_backup.py")

        print(f"\n{'='*60}")
        print(f"🚀 开始实验: {experiment_name}")
        print(f"📁 输出目录: {exp_dir}")
        print(f"{'='*60}\n")

    # DDP: 同步，确保所有进程都能看到实验目录
    if use_ddp:
        dist.barrier()

    writer = SummaryWriter(log_dir=str(log_dir)) if is_main else None

    # ========== 3. 数据准备 ==========
    if is_main:
        print("📦 加载数据集...")

    batch_size = config_override.get('batch_size', cfg.BATCH_SIZE)
    # batch_size 是每个 GPU 的 batch，总 batch = batch_size * world_size
    num_workers = config_override.get('num_workers', cfg.NUM_WORKERS)

    train_dataset = HowlingDataset(
        clean_dir=cfg.TRAIN_CLEAN_DIR,
        howling_dir=cfg.TRAIN_NOISY_DIR,
    )

    # DDP 使用 DistributedSampler，非 DDP 使用普通 shuffle
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # 使用 sampler 时不 shuffle
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )

    val_dataset = HowlingDataset(
        clean_dir=cfg.VAL_CLEAN_DIR,
        howling_dir=cfg.VAL_NOISY_DIR
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if use_ddp else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    if is_main:
        num_gpus = world_size if use_ddp else (len(gpu_ids) if gpu_ids and len(gpu_ids) > 1 else 1)
        effective_batch = batch_size * num_gpus
        per_gpu_batch = batch_size // num_gpus if not use_ddp and num_gpus > 1 else batch_size
        print(f"✅ 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
        print(f"📊 每 GPU batch: {per_gpu_batch}, 总 batch: {effective_batch}, GPUs: {num_gpus}")

    # ========== 4. 模型初始化 ==========
    if is_main:
        print(f"💻 设备: {device}")

    model_class = get_model_class(model_name)
    model = model_class().to(device)

    # 多 GPU 并行训练
    if use_ddp:
        # DDP 模式（通过 torchrun 启动）
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            print(f"🔧 模型: {model.module.__class__.__name__} (DDP, {world_size} GPUs)")
    elif gpu_ids is not None and len(gpu_ids) > 1:
        # DataParallel 模式（通过 --gpus 指定）
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"🔧 模型: {model.module.__class__.__name__} (DataParallel, {len(gpu_ids)} GPUs)")
    else:
        if is_main:
            print(f"🔧 模型: {model.__class__.__name__}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"📊 参数量: {total_params:,}\n")

    # ========== 5. 损失函数和优化器 ==========
    loss_type = config_override.get('loss_function', cfg.DEFAULT_LOSS)
    loss_weights = config_override.get('loss_weights', None)
    criterion = get_loss_function(loss_type, loss_weights)
    if is_main:
        print(f"📉 损失函数: {cfg.LOSS_FUNCTIONS.get(loss_type, loss_type)}")

    learning_rate = config_override.get('learning_rate', cfg.LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 学习率调度器
    lr_scheduler_type = config_override.get('lr_scheduler', cfg.LR_SCHEDULER_TYPE)
    num_epochs = config_override.get('num_epochs', cfg.NUM_EPOCHS)

    scheduler = get_scheduler(
        optimizer,
        lr_scheduler_type,
        num_epochs,
        warmup_epochs=config_override.get('warmup_epochs', cfg.WARMUP_EPOCHS)
    )
    if is_main:
        print(f"📈 学习率调度: {lr_scheduler_type}")

    # 获取实际模型（处理 DDP / DataParallel 包装）
    raw_model = model.module if isinstance(model, (DDP, nn.DataParallel)) else model

    # ========== 6. 混合精度训练 ==========
    use_amp = config_override.get('mixed_precision', cfg.USE_MIXED_PRECISION)
    scaler = GradScaler() if use_amp else None
    if use_amp and is_main:
        print("⚡ 混合精度训练: 已启用")

    # ========== 7. 早停 ==========
    early_stop_patience = args.early_stop
    early_stopping = EarlyStopping(patience=early_stop_patience, min_delta=args.min_delta) if early_stop_patience else None

    # ========== 8. 记录超参数（仅主进程） ==========
    # 统一 GPU 数量计算（与显示逻辑一致）
    num_gpus = world_size if use_ddp else (len(gpu_ids) if gpu_ids and len(gpu_ids) > 1 else 1)
    config_dict = {
        "experiment_name": experiment_name,
        "model": raw_model.__class__.__name__,
        "total_params": total_params,
        "learning_rate": learning_rate,
        "batch_size_per_gpu": batch_size // num_gpus if not use_ddp and num_gpus > 1 else batch_size,
        "total_batch_size": batch_size * num_gpus,
        "num_gpus": num_gpus,
        "world_size": world_size,
        "num_epochs": num_epochs,
        "loss_function": loss_type,
        "lr_scheduler": lr_scheduler_type,
        "mixed_precision": use_amp,
        "parallel_mode": "DDP" if use_ddp else ("DataParallel" if gpu_ids and len(gpu_ids) > 1 else "single"),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    if is_main:
        import json

        with open(exp_dir / "config.json", "w", encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        writer.add_text("Experiment/Config", str(config_dict), 0)

    # ========== 10. 训练循环 ==========
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []

    if is_main:
        print(f"\n{'='*60}")
        print(f"🏋️ 开始训练 ({num_epochs} epochs)...")
        print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        start_time = time.time()

        # DDP: 设置 epoch 以确保每轮数据打乱顺序不同
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            use_amp=use_amp
        )
        train_losses.append(train_loss)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # DDP: 汇总所有进程的 loss
        if use_ddp:
            loss_tensor = torch.tensor([train_loss, val_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = loss_tensor[0].item() / world_size
            val_loss = loss_tensor[1].item() / world_size

        # 更新学习率
        if lr_scheduler_type == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        duration = time.time() - start_time

        # 打印进度（仅主进程）
        if is_main:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {duration:.1f}s")

        # TensorBoard记录（仅主进程）
        if is_main and writer is not None:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Training/LR", current_lr, epoch)
            writer.add_scalar("Time/Epoch", duration, epoch)

        # 保存最佳模型（仅主进程）
        if is_main and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "config": config_dict,
            }

            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            print(f"  ✨ 新的最佳模型 (Val Loss: {val_loss:.4f})")

        # 可视化（每5个epoch，仅主进程）
        if is_main and writer is not None and epoch % 5 == 0:
            with torch.no_grad():
                sample_noisy, sample_clean = next(iter(val_loader))
                sample_noisy = sample_noisy[0:1].to(device)
                sample_pred = model(sample_noisy)

                writer.add_image("Spectrogram/Noisy", sample_noisy[0], epoch, dataformats="CHW")
                writer.add_image("Spectrogram/Pred", sample_pred[0], epoch, dataformats="CHW")
                writer.add_image("Spectrogram/Clean", sample_clean[0:1].to(device)[0], epoch, dataformats="CHW")

        # 早停检查
        if early_stopping is not None:
            if early_stopping(val_loss):
                if is_main:
                    print(f"\n⏹️ 早停触发 (Epoch {epoch+1})")
                break

    # 训练完成
    if is_main:
        print(f"\n{'='*60}")
        print(f"✅ 训练完成!")
        print(f"🏆 最佳验证Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
        print(f"💾 模型保存至: {checkpoint_dir / 'best_model.pth'}")
        print(f"{'='*60}")

        if writer is not None:
            writer.close()

    # 清理 DDP
    if use_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()