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


class SISDRLoss(nn.Module):
    """尺度不变信噪比损失 (SI-SDR Loss)

    基于频谱域的SI-SDR计算，负值作为损失（越小越好）
    """

    def __init__(self, eps: float = 1e-8):
        super(SISDRLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 将预测对齐到目标尺度
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        # s_target = (<target, pred> / ||target||^2) * target
        dot = torch.sum(pred_flat * target_flat, dim=1, keepdim=True)
        s_target_energy = torch.sum(target_flat ** 2, dim=1, keepdim=True) + self.eps
        s_target = (dot / s_target_energy) * target_flat

        # e_noise = pred - s_target
        e_noise = pred_flat - s_target

        # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        si_sdr = 10 * torch.log10(
            torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + self.eps) + self.eps
        )

        # 返回负SI-SDR作为损失
        return -si_sdr.mean()


class MultiResolutionSTFTLoss(nn.Module):
    """多分辨率STFT损失

    在不同分辨率下计算频谱L1损失（帧长512/256/128）
    由于数据已经是频谱域，通过不同粒度的池化模拟多分辨率
    """

    def __init__(self, eps: float = 1e-8):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.eps = eps

    def _compute_loss_at_scale(self, pred: torch.Tensor, target: torch.Tensor,
                                freq_scale: int, time_scale: int) -> torch.Tensor:
        """在指定缩放下计算L1损失"""
        if freq_scale > 1 or time_scale > 1:
            # 使用平均池化模拟不同分辨率
            B, C, F, T = pred.shape
            f_out = F // freq_scale
            t_out = T // time_scale
            if f_out > 0 and t_out > 0:
                pred_scaled = nn.functional.avg_pool2d(pred, (freq_scale, time_scale))
                target_scaled = nn.functional.avg_pool2d(target, (freq_scale, time_scale))
            else:
                pred_scaled = pred
                target_scaled = target
        else:
            pred_scaled = pred
            target_scaled = target
        return nn.functional.l1_loss(pred_scaled, target_scaled)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 确保输入是4D: [B, C, F, T]
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        # 原始分辨率
        loss_full = self._compute_loss_at_scale(pred, target, 1, 1)
        # 频率减半（模拟帧长256）
        loss_half_f = self._compute_loss_at_scale(pred, target, 2, 1)
        # 频率1/4（模拟帧长128）
        loss_quarter_f = self._compute_loss_at_scale(pred, target, 4, 1)
        # 时间减半
        loss_half_t = self._compute_loss_at_scale(pred, target, 1, 2)

        return loss_full + loss_half_f + loss_quarter_f + loss_half_t


class CompositeLoss(nn.Module):
    """组合损失 (Composite Loss)

    多分辨率STFT损失(α=1.0) + SI-SDR损失(β=0.5)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super(CompositeLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mrstft_loss = MultiResolutionSTFTLoss()
        self.sisdr_loss = SISDRLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_mr = self.mrstft_loss(pred, target)
        loss_si = self.sisdr_loss(pred, target)
        return self.alpha * loss_mr + self.beta * loss_si


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
