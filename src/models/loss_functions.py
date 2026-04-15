"""损失函数模块

音频啸叫抑制模型的损失函数实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    """频谱损失
    
    基于对数域幅度谱距离的损失
    """
    
    def __init__(self):
        """初始化频谱损失"""
        super(SpectralLoss, self).__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算频谱损失"""
        epsilon = 1e-8
        
        pred_log = torch.log10(pred + epsilon)
        target_log = torch.log10(target + epsilon)
        
        loss = F.l1_loss(pred_log, target_log)
        return loss


class SpectralConsistencyLoss(nn.Module):
    """频谱一致性损失
    
    确保频谱平滑和连贯，减少伪影
    """
    
    def __init__(self, lambda_freq: float = 0.1, lambda_time: float = 0.1):
        """初始化频谱一致性损失"""
        super(SpectralConsistencyLoss, self).__init__()
        self.lambda_freq = lambda_freq
        self.lambda_time = lambda_time
    
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """计算频谱一致性损失"""
        # 频率梯度
        freq_grad = torch.diff(spectrogram, dim=2)
        freq_loss = torch.mean(torch.abs(freq_grad))
        
        # 时间梯度
        time_grad = torch.diff(spectrogram, dim=3)
        time_loss = torch.mean(torch.abs(time_grad))
        
        total_loss = self.lambda_freq * freq_loss + self.lambda_time * time_loss
        
        return total_loss


class MultiTaskLoss(nn.Module):
    """多任务损失
    
    结合频谱、L1、MSE和一致性损失
    """
    
    def __init__(
        self,
        weights: dict = None,
        use_spectral: bool = True,
        use_l1: bool = True,
        use_mse: bool = True,
        use_consistency: bool = False
    ):
        """初始化多任务损失"""
        super(MultiTaskLoss, self).__init__()
        
        if weights is None:
            weights = {
                'spectral': 0.5,
                'l1': 0.3,
                'mse': 0.2,
                'consistency': 0.0
            }
        
        self.weights = weights
        self.use_spectral = use_spectral
        self.use_l1 = use_l1
        self.use_mse = use_mse
        self.use_consistency = use_consistency
        
        self.spectral_loss = SpectralLoss() if use_spectral else None
        self.l1_loss = nn.L1Loss() if use_l1 else None
        self.mse_loss = nn.MSELoss() if use_mse else None
        self.consistency_loss = SpectralConsistencyLoss() if use_consistency else None
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """计算多任务损失"""
        total_loss = 0.0
        loss_dict = {}
        
        # 频谱损失
        if self.use_spectral and self.spectral_loss is not None:
            spec_loss = self.spectral_loss(pred, target)
            total_loss += self.weights['spectral'] * spec_loss
            loss_dict['spectral'] = spec_loss.item()
        
        # L1损失
        if self.use_l1 and self.l1_loss is not None:
            l1_loss = self.l1_loss(pred, target)
            total_loss += self.weights['l1'] * l1_loss
            loss_dict['l1'] = l1_loss.item()
        
        # MSE损失
        if self.use_mse and self.mse_loss is not None:
            mse_loss = self.mse_loss(pred, target)
            total_loss += self.weights['mse'] * mse_loss
            loss_dict['mse'] = mse_loss.item()
        
        # 一致性损失
        if self.use_consistency and self.consistency_loss is not None:
            cons_loss = self.consistency_loss(pred)
            total_loss += self.weights['consistency'] * cons_loss
            loss_dict['consistency'] = cons_loss.item()
        
        return total_loss, loss_dict


class Discriminator(nn.Module):
    """判别器
    
    GAN训练用的判别网络
    """
    
    def __init__(self, input_channels: int = 1):
        """初始化判别器"""
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1),
            # 注意：不使用 Sigmoid，配合 BCEWithLogitsLoss 以兼容 autocast
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.conv_layers(x)
        output = self.final_layer(features)
        return output


class SpectralConvergenceLoss(nn.Module):
    """频谱收敛损失（Spectral Convergence Loss）

    计算预测频谱与目标频谱之间的相对Frobenius范数误差。
    SC = ||S - Ŝ||_F / ||S||_F
    适用于频谱域模型，衡量频谱整体形态的匹配程度。
    """

    def __init__(self, epsilon: float = 1e-8):
        super(SpectralConvergenceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Frobenius范数：对所有频谱维度求平方和再开根号
        numerator = torch.norm(pred - target, p='fro', dim=(-2, -1))
        denominator = torch.norm(target, p='fro', dim=(-2, -1)) + self.epsilon
        sc_loss = numerator / denominator
        return sc_loss.mean()


class MultiResolutionSTFTLoss(nn.Module):
    """多分辨率STFT损失

    在不同频率分辨率下计算频谱L1损失
    模拟不同FFT帧长（512/256/128）的频谱分析效果
    """

    def __init__(self, fft_sizes: list = None, epsilon: float = 1e-8):
        super(MultiResolutionSTFTLoss, self).__init__()
        # 模拟不同FFT帧长对应的下采样因子
        # fft=512 → 原始分辨率, fft=256 → 2x下采样, fft=128 → 4x下采样
        self.downsample_factors = [1, 2, 4] if fft_sizes is None else \
            [512 // fs for fs in fft_sizes]
        self.epsilon = epsilon

    def _downsample_spectrogram(self, spec: torch.Tensor, factor: int) -> torch.Tensor:
        """对频谱图进行频率维度下采样，模拟更小的FFT帧长"""
        if factor == 1:
            return spec
        freq_dim = spec.shape[-2]
        # 截断到可整除的长度
        trunc_len = (freq_dim // factor) * factor
        spec_trunc = spec[..., :trunc_len, :]
        # 重塑并取均值
        new_shape = list(spec_trunc.shape)
        new_shape[-2] = trunc_len // factor
        new_shape.insert(-1, factor)
        return spec_trunc.reshape(new_shape).mean(dim=-2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for factor in self.downsample_factors:
            pred_down = self._downsample_spectrogram(pred, factor)
            target_down = self._downsample_spectrogram(target, factor)
            total_loss += F.l1_loss(pred_down, target_down)

        return total_loss / len(self.downsample_factors)


class CompositeLoss(nn.Module):
    """复合损失函数

    组合多分辨率STFT损失和频谱收敛损失
    Composite = α * MultiResolutionSTFT + β * SpectralConvergence
    默认: α=1.0, β=0.5
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super(CompositeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mrstft_loss = MultiResolutionSTFTLoss()
        self.sc_loss = SpectralConvergenceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mrstft = self.mrstft_loss(pred, target)
        sc = self.sc_loss(pred, target)
        return self.alpha * mrstft + self.beta * sc


class AdversarialLoss(nn.Module):
    """对抗损失

    GAN训练的生成器和判别器损失
    """
    
    def __init__(self, loss_type: str = 'lsgan'):
        """初始化对抗损失"""
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'standard':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'wgan':
            self.criterion = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """计算生成器损失"""
        if self.loss_type == 'standard':
            target = torch.ones_like(fake_pred)
            return self.criterion(fake_pred, target)
        elif self.loss_type == 'lsgan':
            target = torch.ones_like(fake_pred)
            return self.criterion(fake_pred, target)
        elif self.loss_type == 'wgan':
            return -torch.mean(fake_pred)
    
    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """计算判别器损失"""
        if self.loss_type == 'standard':
            target_real = torch.ones_like(real_pred)
            loss_real = self.criterion(real_pred, target_real)
            
            target_fake = torch.zeros_like(fake_pred)
            loss_fake = self.criterion(fake_pred, target_fake)
            
            return (loss_real + loss_fake) / 2
        
        elif self.loss_type == 'lsgan':
            target_real = torch.ones_like(real_pred)
            loss_real = self.criterion(real_pred, target_real)
            
            target_fake = torch.zeros_like(fake_pred)
            loss_fake = self.criterion(fake_pred, target_fake)
            
            return (loss_real + loss_fake) / 2
        
        elif self.loss_type == 'wgan':
            return -torch.mean(real_pred) + torch.mean(fake_pred)
