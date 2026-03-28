"""训练策略模块

音频啸叫抑制模型的训练策略实现，包括：
- 混合精度训练
- 高级学习率调度
- 课程学习
"""

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import numpy as np
from typing import Dict, List, Optional, Tuple


class MixedPrecisionTrainer:
    """混合精度训练器
    
    使用PyTorch AMP（Automatic Mixed Precision）加速训练
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_amp: bool = True
    ):
        """初始化混合精度训练器
        
        Args:
            model: 模型
            optimizer: 优化器
            device: 设备
            use_amp: 是否使用混合精度
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        
        # 初始化GradScaler
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ 混合精度训练已启用 (AMP)")
        else:
            self.scaler = None
            print("ℹ️ 使用标准精度训练")
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        model: Optional[nn.Module] = None
    ) -> Tuple[float, torch.Tensor]:
        """执行一步训练
        
        Args:
            inputs: 输入数据
            targets: 目标数据
            criterion: 损失函数
            model: 可选，如果不提供则使用self.model
            
        Returns:
            (loss_value, predictions)
        """
        model_to_use = model if model is not None else self.model
        
        self.optimizer.zero_grad()
        
        if self.use_amp and self.scaler is not None:
            # 混合精度训练
            with torch.cuda.amp.autocast():
                predictions = model_to_use(inputs)
                loss = criterion(predictions, targets)
            
            # 反向传播和参数更新
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 标准精度训练
            predictions = model_to_use(inputs)
            loss = criterion(predictions, targets)
            
            loss.backward()
            self.optimizer.step()
        
        return loss.item(), predictions
    
    def get_state(self) -> dict:
        """获取训练器状态（用于保存checkpoint）"""
        state = {'use_amp': self.use_amp}
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        return state
    
    def load_state(self, state: dict):
        """加载训练器状态"""
        if 'scaler' in state and self.scaler is not None:
            self.scaler.load_state_dict(state['scaler'])


class CosineAnnealingWarmupScheduler:
    """带Warmup的Cosine Annealing学习率调度器
    
    结合warmup和cosine annealing的优点
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6
    ):
        """初始化调度器
        
        Args:
            optimizer: 优化器
            warmup_epochs: warmup轮数
            total_epochs: 总训练轮数
            base_lr: 基础学习率
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        print(f"✅ Cosine Annealing + Warmup调度器已初始化")
        print(f"   Warmup: {warmup_epochs} epochs")
        print(f"   Base LR: {base_lr}, Min LR: {min_lr}")
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup阶段：线性增长
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine Annealing阶段
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * (
                0.5 * (1 + np.cos(np.pi * progress))
            )
        
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class OneCycleScheduler:
    """One Cycle学习率调度器
    
    按照super-convergence策略调整学习率
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_epochs: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4
    ):
        """初始化调度器
        
        Args:
            optimizer: 优化器
            max_lr: 最大学习率
            total_epochs: 总训练轮数
            pct_start: 上升阶段占比（0-1）
            div_factor: 初始学习率 = max_lr / div_factor
            final_div_factor: 最终学习率 = max_lr / final_div_factor
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.current_epoch = 0
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        print(f"✅ One Cycle调度器已初始化")
        print(f"   Max LR: {max_lr}")
        print(f"   Initial LR: {self.initial_lr}")
        print(f"   Final LR: {self.final_lr}")
        print(f"   Rise epochs: {int(total_epochs * pct_start)}")
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        step = self.current_epoch - 1
        total_steps = self.total_epochs
        
        if step < total_steps * self.pct_start:
            # 上升阶段
            scale = step / (total_steps * self.pct_start)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * scale
        else:
            # 下降阶段
            scale = (step - total_steps * self.pct_start) / (
                total_steps * (1 - self.pct_start)
            )
            lr = self.max_lr + (self.final_lr - self.max_lr) * scale
        
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class CurriculumLearning:
    """课程学习策略
    
    从简单到复杂逐步增加训练难度
    """
    
    def __init__(
        self,
        total_epochs: int,
        difficulty_levels: List[Dict[str, float]],
        schedule_type: str = 'step'
    ):
        """初始化课程学习
        
        Args:
            total_epochs: 总训练轮数
            difficulty_levels: 难度级别列表，每个级别是一个字典
                例如: [{'noise_level': 0.01}, {'noise_level': 0.05}]
            schedule_type: 调度类型 ('step', 'linear', 'exponential')
        """
        self.total_epochs = total_epochs
        self.difficulty_levels = difficulty_levels
        self.schedule_type = schedule_type
        self.num_levels = len(difficulty_levels)
        
        print(f"✅ 课程学习策略已初始化")
        print(f"   难度级别数: {self.num_levels}")
        print(f"   调度类型: {schedule_type}")
        print(f"   难度级别:")
        for i, level in enumerate(difficulty_levels):
            print(f"     Level {i+1}: {level}")
    
    def get_current_difficulty(self, epoch: int) -> Dict[str, float]:
        """获取当前epoch的难度级别
        
        Args:
            epoch: 当前epoch
            
        Returns:
            难度参数字典
        """
        if self.schedule_type == 'step':
            # 阶梯式：均匀分配
            level_idx = min(
                int(epoch / self.total_epochs * self.num_levels),
                self.num_levels - 1
            )
            return self.difficulty_levels[level_idx]
        
        elif self.schedule_type == 'linear':
            # 线性插值
            progress = epoch / self.total_epochs
            level_idx_float = progress * (self.num_levels - 1)
            level_idx_0 = int(level_idx_float)
            level_idx_1 = min(level_idx_0 + 1, self.num_levels - 1)
            alpha = level_idx_float - level_idx_0
            
            difficulty = {}
            for key in self.difficulty_levels[0].keys():
                val_0 = self.difficulty_levels[level_idx_0][key]
                val_1 = self.difficulty_levels[level_idx_1][key]
                difficulty[key] = val_0 + alpha * (val_1 - val_0)
            
            return difficulty
        
        elif self.schedule_type == 'exponential':
            # 指数增长
            progress = epoch / self.total_epochs
            level_idx = min(
                int((progress ** 2) * self.num_levels),
                self.num_levels - 1
            )
            return self.difficulty_levels[level_idx]
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def get_difficulty_description(self, epoch: int) -> str:
        """获取当前难度的描述"""
        difficulty = self.get_current_difficulty(epoch)
        desc = ", ".join([f"{k}={v:.3f}" for k, v in difficulty.items()])
        return desc


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    **kwargs
) -> object:
    """创建学习率调度器工厂函数
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('cosine_warmup', 'one_cycle', 'plateau', 'step')
        **kwargs: 调度器特定参数
        
    Returns:
        学习率调度器实例
    """
    if scheduler_type == 'cosine_warmup':
        return CosineAnnealingWarmupScheduler(
            optimizer=optimizer,
            **kwargs
        )
    
    elif scheduler_type == 'one_cycle':
        return OneCycleScheduler(
            optimizer=optimizer,
            **kwargs
        )
    
    elif scheduler_type == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 3),
            verbose=True
        )
    
    elif scheduler_type == 'step':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
