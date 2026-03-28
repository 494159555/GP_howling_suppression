"""音频啸叫抑制模型模块

可用模型:
- AudioUNet3: 3层轻量级U-Net
- AudioUNet5: 5层标准U-Net（默认模型）
- AudioUNet5Attention: 5层U-Net + 注意力门
- AudioUNet5Optimized: 5层U-Net + 注意力+残差+空洞（综合优化）
- AudioUNet5GAN: 5层U-Net + GAN框架
"""

from .unet_v1 import AudioUNet3
from .unet_v2 import AudioUNet5
from .unet_v3_attention import AudioUNet5Attention
from .unet_v6_optimized import AudioUNet5Optimized
from .unet_v10_gan import AudioUNet5GAN

from .loss_functions import (
    SpectralLoss,
    SpectralConsistencyLoss,
    MultiTaskLoss,
    AdversarialLoss,
    Discriminator,
)
from .augmentation import (
    AudioAugmentation,
    SpecAugment,
    MixupAugmentation,
    AdversarialAugmentation,
    CombinedAugmentation,
)
from .training_strategies import (
    MixedPrecisionTrainer,
    CosineAnnealingWarmupScheduler,
    OneCycleScheduler,
    CurriculumLearning,
    create_lr_scheduler,
)
from .post_processing import (
    AdaptivePostProcessing,
    MultiFrameSmoother,
    AdaptiveGainControl,
    PostProcessingPipeline,
)

__all__ = [
    # 模型
    'AudioUNet3', 'AudioUNet5', 'AudioUNet5Attention',
    'AudioUNet5Optimized', 'AudioUNet5GAN',
    # 损失函数
    'SpectralLoss', 'SpectralConsistencyLoss', 'MultiTaskLoss',
    'AdversarialLoss', 'Discriminator',
    # 数据增强
    'AudioAugmentation', 'SpecAugment', 'MixupAugmentation',
    'AdversarialAugmentation', 'CombinedAugmentation',
    # 训练策略
    'MixedPrecisionTrainer', 'CosineAnnealingWarmupScheduler',
    'OneCycleScheduler', 'CurriculumLearning', 'create_lr_scheduler',
    # 后处理
    'AdaptivePostProcessing', 'MultiFrameSmoother',
    'AdaptiveGainControl', 'PostProcessingPipeline',
    # 工具
    'MODEL_CLASSES', 'get_model', 'list_models',
]

MODEL_CLASSES = {
    'unet_v1': AudioUNet3,
    'unet_v2': AudioUNet5,
    'unet_v3_attention': AudioUNet5Attention,
    'unet_v6_optimized': AudioUNet5Optimized,
    'unet_v10_gan': AudioUNet5GAN,
}


def get_model(model_name: str):
    """通过模型名称获取模型类"""
    if model_name not in MODEL_CLASSES:
        available = ', '.join(MODEL_CLASSES.keys())
        raise ValueError(f"未知模型: {model_name}. 可用模型: {available}")
    return MODEL_CLASSES[model_name]


def list_models():
    """列出所有可用模型"""
    return list(MODEL_CLASSES.keys())
