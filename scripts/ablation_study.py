"""实验3: 消融实验（注意力/残差/空洞贡献分析）

基于AudioUNet5Optimized架构进行消融实验，分析各组件贡献：
- 基线: 无任何增强组件的5层U-Net
- +注意力: 仅添加注意力门
- +残差: 仅添加残差连接
- +空洞: 仅添加空洞卷积
- +注意力+残差: 注意力+残差组合
- +注意力+空洞: 注意力+空洞组合
- +残差+空洞: 残差+空洞组合
- 完整版: 注意力+残差+空洞（全部启用）

用法:
    python scripts/ablation_study.py
    python scripts/ablation_study.py --epochs 50 --debug
    python scripts/ablation_study.py --components attention residual
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import HowlingDataset
from src.models.blocks import make_encoder_block, make_decoder_block, make_output_block
from src.models.modules.attention_modules import AttentionBlock, ResidualBlock, AtrousConvBlock
from src.models.loss_functions import SpectralLoss
from src.evaluation.metrics import AudioMetrics, calculate_mos_score


class AblationUNet(nn.Module):
    """可配置的消融U-Net，支持开关各组件"""

    def __init__(self, use_attention=False, use_residual=False, use_dilated=False,
                 dilation_rates=None):
        super().__init__()
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_dilated = use_dilated
        self.dilation_rates = dilation_rates or [2, 4, 8]

        # 编码器
        self.enc1_down = make_encoder_block(1, 16)
        self.enc2_down = make_encoder_block(16, 32)
        self.enc3_down = make_encoder_block(32, 64)
        self.enc4_down = make_encoder_block(64, 128)
        self.enc5_down = make_encoder_block(128, 256)

        # 残差块（可选）
        if use_residual:
            self.res1 = ResidualBlock(16)
            self.res2 = ResidualBlock(32)
            self.res3 = ResidualBlock(64)
            self.res4 = ResidualBlock(128)
            self.res5 = ResidualBlock(256)

        # 瓶颈层空洞卷积（可选）
        if use_dilated:
            self.atrous_block = AtrousConvBlock(256, 256, dilation_rates=self.dilation_rates)

        # 解码器
        self.dec5 = make_decoder_block(256, 128)
        self.dec4 = make_decoder_block(256, 64)
        self.dec3 = make_decoder_block(128, 32)
        self.dec2 = make_decoder_block(64, 16)
        self.dec1 = make_output_block(32)

        # 注意力门（可选）
        if use_attention:
            self.att4 = AttentionBlock(F_g=128, F_l=128, F_int=64)
            self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=32)
            self.att2 = AttentionBlock(F_g=32, F_l=32, F_int=16)
            self.att1 = AttentionBlock(F_g=16, F_l=16, F_int=8)

    def _enc(self, x, enc_layer, idx):
        out = enc_layer(x)
        if self.use_residual:
            res_block = getattr(self, f'res{idx}')
            out = res_block(out)
        return out

    def forward(self, x):
        x_log = torch.log10(x + 1e-8)

        e1 = self._enc(x_log, self.enc1_down, 1)
        e2 = self._enc(e1, self.enc2_down, 2)
        e3 = self._enc(e2, self.enc3_down, 3)
        e4 = self._enc(e3, self.enc4_down, 4)
        e5 = self._enc(e4, self.enc5_down, 5)

        if self.use_dilated:
            e5 = self.atrous_block(e5)

        d5 = self.dec5(e5)
        if self.use_attention:
            e4 = self.att4(d5, e4)
        d5_cat = torch.cat([d5, e4], dim=1)

        d4 = self.dec4(d5_cat)
        if self.use_attention:
            e3 = self.att3(d4, e3)
        d4_cat = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4_cat)
        if self.use_attention:
            e2 = self.att2(d3, e2)
        d3_cat = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3_cat)
        if self.use_attention:
            e1 = self.att1(d2, e1)
        d2_cat = torch.cat([d2, e1], dim=1)

        mask = self.dec1(d2_cat)
        return x * mask


# 消融配置: (名称, use_attention, use_residual, use_dilated)
ABLATION_CONFIGS = [
    ('baseline',          False, False, False),
    ('+attention',        True,  False, False),
    ('+residual',         False, True,  False),
    ('+dilated',          False, False, True),
    ('+attn+res',         True,  True,  False),
    ('+attn+dilated',     True,  False, True),
    ('+res+dilated',      False, True,  True),
    ('full(+attn+res+dil)', True, True,  True),
]


def train_and_evaluate(config_name, use_attention, use_residual, use_dilated,
                       train_loader, val_loader, val_loader_td, device, args):
    """训练并评估单个消融配置"""
    print(f"\n{'='*60}")
    print(f"  消融配置: {config_name}")
    print(f"  注意力={use_attention}, 残差={use_residual}, 空洞={use_dilated}")
    print(f"{'='*60}")

    model = AblationUNet(
        use_attention=use_attention,
        use_residual=use_residual,
        use_dilated=use_dilated,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {param_count:,}")

    criterion = SpectralLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    num_epochs = args.epochs
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # 训练
        model.train()
        epoch_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
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
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1:3d}/{num_epochs}] "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # 最终评估
    model.eval()
    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)
    all_metrics = {
        'snr_improvement_db': [], 'psnr_db': [], 'stoi_score': [],
        'howling_reduction_db': [],
    }
    final_losses = []

    # 优先使用时域评估（STOI/PESQ需要时域信号）
    use_td = val_loader_td is not None
    with torch.no_grad():
        td_iter = iter(val_loader_td) if use_td else None
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            final_losses.append(torch.nn.L1Loss()(pred, clean).item())

            if use_td:
                try:
                    td_batch = next(td_iter)
                    td_noisy, td_clean, td_noisy_wave, td_clean_wave, td_noisy_stft = td_batch
                    td_noisy = td_noisy.to(device)
                    td_pred = model(td_noisy)
                    from src.evaluate import _istft_from_mag_phase
                    enhanced_wave = _istft_from_mag_phase(td_pred.cpu(), td_noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                    noisy_wave_td = _istft_from_mag_phase(td_noisy.cpu(), td_noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                    m = metrics_calc.calculate_all_metrics(clean=td_clean_wave, noisy=noisy_wave_td, enhanced=enhanced_wave)
                except Exception:
                    m = metrics_calc.calculate_all_metrics(clean=clean, noisy=noisy, enhanced=pred)
            else:
                m = metrics_calc.calculate_all_metrics(clean=clean, noisy=noisy, enhanced=pred)

            for k in all_metrics:
                if k in m:
                    all_metrics[k].append(m[k])

    results = {
        'config': config_name,
        'use_attention': use_attention,
        'use_residual': use_residual,
        'use_dilated': use_dilated,
        'param_count': param_count,
        'best_val_loss': best_val_loss,
        'avg_loss': float(np.mean(final_losses)),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    for k, v in all_metrics.items():
        results[k] = float(np.mean(v)) if v else 0.0
    results['mos_estimate'] = calculate_mos_score(results)

    print(f"\n  结果: Loss={results['avg_loss']:.4f}, "
          f"SNR={results['snr_improvement_db']:.2f}dB, "
          f"MOS={results['mos_estimate']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='实验3: 消融实验')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='experiments/exp3_ablation',
                        help='输出目录')
    parser.add_argument('--debug', action='store_true', help='调试模式（3 epochs）')
    parser.add_argument('--components', nargs='+', default=None,
                        choices=['baseline', '+attention', '+residual', '+dilated',
                                 '+attn+res', '+attn+dilated', '+res+dilated', 'full'],
                        help='只运行指定配置的消融实验')
    args = parser.parse_args()

    if args.debug:
        args.epochs = 3

    # 随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = cfg.DEVICE
    print(f"设备: {device}")

    # 数据
    print("\n加载数据集...")
    train_dataset = HowlingDataset(cfg.TRAIN_CLEAN_DIR, cfg.TRAIN_NOISY_DIR)
    val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
    # 带波形的验证集用于时域评估（STOI/PESQ/SI-SDR）
    val_dataset_td = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR, return_waveform=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_loader_td = DataLoader(val_dataset_td, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"训练: {len(train_dataset)}, 验证: {len(val_dataset)}")

    # 输出目录
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 筛选配置
    configs = ABLATION_CONFIGS
    if args.components:
        configs = [c for c in configs if c[0] in args.components]

    print(f"\n消融配置 ({len(configs)}个):")
    for name, attn, res, dil in configs:
        flags = []
        if attn: flags.append('注意力')
        if res: flags.append('残差')
        if dil: flags.append('空洞')
        print(f"  {name:25s} [{', '.join(flags) or '无'}]")

    # 运行消融实验
    all_results = []
    for name, attn, res, dil in configs:
        result = train_and_evaluate(
            name, attn, res, dil, train_loader, val_loader, val_loader_td, device, args
        )
        all_results.append(result)

    # 保存结果
    results_path = output_dir / 'ablation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)

    # 打印对比表格
    print(f"\n{'='*90}")
    print("  消融实验结果")
    print(f"{'='*90}")
    print(f"{'配置':25s} | {'参数量':>10s} | {'Loss':>8s} | {'SNR':>8s} | {'STOI':>7s} | {'MOS':>5s}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: x['mos_estimate'], reverse=True):
        print(f"{r['config']:25s} | {r['param_count']:>10,d} | "
              f"{r['avg_loss']:8.4f} | {r['snr_improvement_db']:8.2f} | "
              f"{r['stoi_score']:7.4f} | {r['mos_estimate']:5.2f}")

    print(f"\n结果已保存: {results_path}")


if __name__ == '__main__':
    main()
