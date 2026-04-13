"""配置模块

定义音频啸叫抑制项目的全局配置参数
"""

from pathlib import Path


class Config:
    """全局配置类"""
    
    # ============ 路径配置 ============
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # DATA_ROOT = PROJECT_ROOT / "data"
    DATA_ROOT=Path("/mnt/ent_disk0/syx/howling_data")
    
    TRAIN_CLEAN_DIR = DATA_ROOT / "train" / "clean"
    TRAIN_NOISY_DIR = DATA_ROOT / "train" / "howling"
    
    VAL_CLEAN_DIR = DATA_ROOT / "dev" / "clean"
    VAL_NOISY_DIR = DATA_ROOT / "dev" / "howling"
    
    EXP_DIR = PROJECT_ROOT / "experiments"
    
    # ============ 音频处理参数 ============
    SAMPLE_RATE = 16000      # 采样率
    CHUNK_LEN = 3.0          # 音频片段长度(秒)
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_LEN)
    N_FFT = 512              # FFT窗口大小
    HOP_LENGTH = 128         # STFT跳跃长度
    
    # ============ 训练超参数 ============
    BATCH_SIZE = 64          # 批大小（多卡训练建议32+）
    LEARNING_RATE = 1e-4     # 学习率
    NUM_EPOCHS = 100          # 训练轮数
    NUM_WORKERS = 16          # 数据加载线程数（建议: GPU数×2~4）
    
    # 学习率调度器参数
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    
    # ============ 模型配置 ============
    # 注意：仅列出实际存在的模型文件
    AVAILABLE_MODELS = {
        'unet_v1': 'AudioUNet3',
        'unet_v2': 'AudioUNet5',
        'unet_v3_attention': 'AudioUNet5Attention',
        'unet_v6_optimized': 'AudioUNet5Optimized',
        'unet_v10_gan': 'AudioUNet5GAN',
    }

    DEFAULT_MODEL = 'unet_v2'

    MODEL_DESCRIPTIONS = {
        'unet_v1': '3层U-Net (轻量级基线)',
        'unet_v2': '5层U-Net (默认模型)',
        'unet_v3_attention': '5层U-Net + 注意力门',
        'unet_v6_optimized': '5层U-Net + 注意力+残差+空洞 (综合优化)',
        'unet_v10_gan': '5层U-Net + GAN框架',
    }
    
    LOSS_FUNCTIONS = {
        'l1': 'L1损失',
        'mse': 'MSE损失',
        'spectral': '频谱损失',
        'multitask': '多任务损失',
        'multitask_consistency': '多任务一致性损失',
        'adversarial': '对抗损失',
    }
    
    DEFAULT_LOSS = 'multitask'

    # ============ 训练策略配置 ============
    USE_MIXED_PRECISION = True   # 是否使用混合精度训练（RTX 3090建议开启）
    CURRICULUM_LEARNING = False  # 是否使用课程学习
    LR_SCHEDULER_TYPE = 'plateau'  # 学习率调度器类型: 'plateau', 'cosine_warmup', 'one_cycle', 'step'
    
    # 混合精度训练参数
    AMP_ENABLED = True  # AMP是否启用
    
    # 课程学习参数
    CURRICULUM_TOTAL_EPOCHS = 50  # 总训练轮数
    CURRICULUM_DIFFICULTY_LEVELS = [  # 难度级别
        {'noise_level': 0.01},
        {'noise_level': 0.05},
        {'noise_level': 0.1},
        {'noise_level': 0.2}
    ]
    CURRICULUM_SCHEDULE_TYPE = 'step'  # 调度类型: 'step', 'linear', 'exponential'
    
    # 学习率调度参数
    WARMUP_EPOCHS = 5  # Warmup轮数
    COSINE_MIN_LR = 1e-6  # Cosine Annealing最小学习率
    ONE_CYCLE_MAX_LR = 1e-3  # One Cycle最大学习率
    ONE_CYCLE_PCT_START = 0.3  # One Cycle上升阶段占比
    
    # ============ 后处理配置 ============
    USE_POST_PROCESSING = False  # 是否使用后处理
    POST_PROCESSING_METHOD = 'pipeline'  # 后处理方法: 'adaptive', 'smoothing', 'gain', 'pipeline'
    
    # 自适应后处理参数
    ADAPTIVE_THRESHOLD = 0.1  # 固定阈值
    ADAPTIVE_THRESHOLD_PERCENTILE = 0.9  # 自适应阈值百分位
    ADAPTIVE_SMOOTHING_WINDOW = 5  # 平滑窗口
    ADAPTIVE_MODE = True  # 是否使用自适应阈值
    
    # 多帧平滑参数
    SMOOTHING_METHOD = 'moving_average'  # 平滑方法: 'moving_average', 'kalman', 'wiener', 'median'
    SMOOTHING_WINDOW_SIZE = 5  # 窗口大小
    KALMAN_Q = 0.01  # Kalman过程噪声
    KALMAN_R = 0.1  # Kalman观测噪声
    
    # 增益控制参数
    GAIN_CONTROL_METHOD = 'agc'  # 增益控制方法: 'agc', 'drc', 'limiter'
    TARGET_LEVEL = 0.7  # 目标电平
    COMPRESSION_RATIO = 4.0  # 压缩比
    ATTACK_TIME = 0.01  # 攻击时间（秒）
    RELEASE_TIME = 0.1  # 释放时间（秒）

    # ============ 设备配置 ============
    @property
    def DEVICE(self):
        """获取计算设备"""
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()