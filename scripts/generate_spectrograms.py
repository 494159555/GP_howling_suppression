"""生成论文频谱对比图（图4-7）

加载一个测试样本，经所有方法处理后，生成一张 7 子图拼接的频谱对比大图：
(a) 干净语音  (b) 带啸叫语音  (c) 移频法  (d) 自适应反馈消除法
(e) AudioUNet3  (f) AudioUNet5Attention  (g) AudioUNet5Optimized

用法:
    python scripts/generate_spectrograms.py
    python scripts/generate_spectrograms.py --sample-idx 5
    python scripts/generate_spectrograms.py --skip-models
    python scripts/generate_spectrograms.py --output-dir experiments/spectrogram_comparison --dpi 300
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import HowlingDataset
from src.models import MODEL_CLASSES, get_model


# ============ 频谱反归一化 ============

NORM_MIN = -11.5
NORM_MAX = 2.5


def denorm_spectrogram(spec_norm):
    """将归一化频谱 [0,1] 反归一化回 log10 域"""
    return spec_norm * (NORM_MAX - NORM_MIN) + NORM_MIN


# ============ 查找检查点 ============

def find_best_checkpoints():
    """扫描 experiments 目录，为每个模型找最佳检查点"""
    exp_dir = PROJECT_ROOT / 'experiments'
    checkpoints = {}

    if not exp_dir.exists():
        return checkpoints

    for exp_path in sorted(exp_dir.iterdir()):
        if not exp_path.is_dir():
            continue
        ckpt_path = exp_path / 'checkpoints' / 'best_model.pth'
        config_path = exp_path / 'config.json'

        if not ckpt_path.exists():
            continue

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                exp_config = json.load(f)
            model_class_name = exp_config.get('model', '')
            for name, cls in MODEL_CLASSES.items():
                if cls.__name__ == model_class_name:
                    if name not in checkpoints:
                        checkpoints[name] = {
                            'path': str(ckpt_path),
                            'exp_name': exp_path.name,
                        }
                    break

    return checkpoints


def load_model(model_name, checkpoint_path, device):
    """加载模型并设为评估模式"""
    model_class = get_model(model_name)
    model = model_class().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# ============ 频谱绘制 ============

def plot_spectrogram_comparison(spectrograms, titles, output_path, dpi=300):
    """绘制频谱对比大图

    Args:
        spectrograms: list of 2D numpy arrays (freq x time)
        titles: list of subplot titles
        output_path: 输出路径
        dpi: 分辨率
    """
    n = len(spectrograms)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.0 * n))

    if n == 1:
        axes = [axes]

    # 统一 colorbar 范围（基于 log 域）
    all_data = np.concatenate([s.flatten() for s in spectrograms])
    vmin, vmax = np.percentile(all_data, [2, 98])

    # 频率轴参数
    n_fft = cfg.N_FFT
    sample_rate = cfg.SAMPLE_RATE
    hop_length = cfg.HOP_LENGTH
    freq_bins = spectrograms[0].shape[0]
    time_frames = spectrograms[0].shape[1]

    freq_axis = np.linspace(0, sample_rate / 2, freq_bins)
    time_axis = np.arange(time_frames) * hop_length / sample_rate

    for i, (spec, title) in enumerate(zip(spectrograms, titles)):
        im = axes[i].imshow(
            spec, aspect='auto', origin='lower',
            cmap='viridis', vmin=vmin, vmax=vmax,
            extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        )
        axes[i].set_title(f'({chr(ord("a") + i)}) {title}', fontsize=13, fontweight='bold', loc='left')
        axes[i].set_ylabel('频率 (Hz)', fontsize=10)
        if i == n - 1:
            axes[i].set_xlabel('时间 (s)', fontsize=10)

        # 只在最后一个子图旁加 colorbar
        if i == n - 1:
            cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02)
            cbar.set_label('幅度 (dB)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"频谱对比图已保存: {output_path}")


def plot_individual_spectrograms(spectrograms, titles, output_dir, dpi=300):
    """额外为每个方法单独生成一张频谱图（备用）

    Args:
        spectrograms: list of 2D numpy arrays
        titles: list of titles
        output_dir: 输出目录
        dpi: 分辨率
    """
    n_fft = cfg.N_FFT
    sample_rate = cfg.SAMPLE_RATE
    hop_length = cfg.HOP_LENGTH

    for spec, title in zip(spectrograms, titles):
        freq_bins = spec.shape[0]
        time_frames = spec.shape[1]
        freq_axis = np.linspace(0, sample_rate / 2, freq_bins)
        time_axis = np.arange(time_frames) * hop_length / sample_rate

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            spec, aspect='auto', origin='lower',
            cmap='viridis',
            extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]],
        )
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('频率 (Hz)', fontsize=11)
        ax.set_xlabel('时间 (s)', fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('幅度 (dB)', fontsize=10)

        safe_name = title.replace(' ', '_').replace('(', '').replace(')', '')
        save_path = os.path.join(output_dir, f'spectrogram_{safe_name}.png')
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  单独频谱图: {save_path}")


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description='生成论文频谱对比图')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='使用测试集中第几个样本（默认0）')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/spectrogram_comparison',
                        help='输出目录')
    parser.add_argument('--dpi', type=int, default=300,
                        help='图片分辨率（默认300）')
    parser.add_argument('--skip-models', action='store_true',
                        help='跳过深度学习模型（仅生成干净/带啸叫/传统方法）')
    parser.add_argument('--individual', action='store_true',
                        help='同时生成各方法的单独频谱图')
    args = parser.parse_args()

    device = cfg.DEVICE
    print(f"设备: {device}")

    # ---- 1. 加载测试数据 ----
    print(f"\n加载测试集...")
    test_dataset = HowlingDataset(
        clean_dir=cfg.TEST_CLEAN_DIR,
        howling_dir=cfg.TEST_NOISY_DIR,
        return_waveform=True,
    )
    print(f"测试样本数: {len(test_dataset)}")

    # 取指定样本
    idx = min(args.sample_idx, len(test_dataset) - 1)
    sample = test_dataset[idx]
    noisy_mag, clean_mag = sample[0], sample[1]  # [1, 256, T]

    # 反归一化到 log 域用于显示
    clean_log = denorm_spectrogram(clean_mag[0].numpy())     # [256, T]
    noisy_log = denorm_spectrogram(noisy_mag[0].numpy())     # [256, T]

    spectrograms = [clean_log, noisy_log]
    titles = ['干净语音', '带啸叫语音']

    # ---- 2. 传统方法 ----
    print("\n处理传统方法...")
    from src.traditional import FrequencyShiftMethod, AdaptiveFeedbackMethod

    noisy_input = noisy_mag.unsqueeze(0).to(device)  # [1, 1, 256, T]

    # 移频法
    fs_method = FrequencyShiftMethod(shift_hz=20.0).to(device)
    fs_method.eval()
    with torch.no_grad():
        fs_output = fs_method(noisy_input)
    fs_log = denorm_spectrogram(fs_output[0, 0].cpu().numpy())
    spectrograms.append(fs_log)
    titles.append('移频法')
    print("  移频法完成")

    # 自适应反馈消除法
    afb_method = AdaptiveFeedbackMethod(filter_length=64).to(device)
    afb_method.eval()
    with torch.no_grad():
        afb_output = afb_method(noisy_input)
    afb_log = denorm_spectrogram(afb_output[0, 0].cpu().numpy())
    spectrograms.append(afb_log)
    titles.append('自适应反馈消除法')
    print("  自适应反馈消除法完成")

    # ---- 3. 深度学习模型 ----
    # 需要生成频谱的模型（按 todolist 中的顺序）
    target_models = {
        'unet_v1': 'AudioUNet3',
        'unet_v3_attention': 'AudioUNet5Attention',
        'unet_v6_optimized': 'AudioUNet5Optimized',
    }

    if not args.skip_models:
        print("\n扫描模型检查点...")
        all_checkpoints = find_best_checkpoints()
        print(f"找到 {len(all_checkpoints)} 个检查点:")

        loaded_models = {}
        for model_name, display_name in target_models.items():
            if model_name in all_checkpoints:
                ckpt_path = all_checkpoints[model_name]['path']
                try:
                    model = load_model(model_name, ckpt_path, device)
                    loaded_models[display_name] = model
                    print(f"  {display_name}: {ckpt_path}")
                except Exception as e:
                    print(f"  {display_name}: 加载失败 - {e}")
            else:
                print(f"  {display_name}: 未找到检查点，跳过")

        # 推理
        for display_name, model in loaded_models.items():
            with torch.no_grad():
                output = model(noisy_input)
            output_log = denorm_spectrogram(output[0, 0].cpu().numpy())
            spectrograms.append(output_log)
            titles.append(display_name)
            print(f"  {display_name} 推理完成")
    else:
        print("\n跳过深度学习模型（--skip-models）")

    # ---- 4. 生成图片 ----
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 合并大图
    combined_path = output_dir / 'figure4_7_spectrogram_comparison.png'
    plot_spectrogram_comparison(spectrograms, titles, str(combined_path), dpi=args.dpi)

    # 单独图（可选）
    if args.individual:
        print("\n生成单独频谱图...")
        plot_individual_spectrograms(spectrograms, titles, str(output_dir), dpi=args.dpi)

    print(f"\n完成！共生成 {len(spectrograms)} 个频谱子图")


if __name__ == '__main__':
    main()
