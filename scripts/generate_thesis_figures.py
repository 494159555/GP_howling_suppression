"""生成论文额外图片：D5 模型性能演进柱状图、D6 训练过程曲线图、消融实验热力图

用法:
    python scripts/generate_thesis_figures.py
    python scripts/generate_thesis_figures.py --output-dir experiments/thesis_figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

# ============ 中文字体设置 ============
def setup_chinese_font():
    """设置matplotlib中文字体"""
    font_candidates = [
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
    ]
    for fp in font_candidates:
        if Path(fp).exists():
            fm.fontManager.addfont(fp)
            prop = fm.FontProperties(fname=fp)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return prop.get_name()
    # fallback
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN']
    plt.rcParams['axes.unicode_minus'] = False
    return None


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============ D5: 模型性能演进柱状图 ============
def generate_performance_bar_chart(output_dir, dpi=300):
    """生成图4-X 模型性能演进对比柱状图（SI-SDR / PESQ / STOI）"""
    eval_path = PROJECT_ROOT / 'experiments/exp2_unified_evaluation/evaluation_results.json'
    if not eval_path.exists():
        print(f"跳过性能柱状图: {eval_path} 不存在")
        return

    data = load_json(eval_path)

    # 模型顺序和显示名称
    model_order = [
        ('FrequencyShift', '移频法'),
        ('GainSuppression', '增益抑制法'),
        ('AdaptiveFeedback', '自适应反馈消除法'),
        ('unet_v1', 'AudioUNet3'),
        ('unet_v2', 'AudioUNet5'),
        ('unet_v3_attention', 'AudioUNet5Attention'),
        ('unet_v6_optimized', 'AudioUNet5Optimized'),
    ]

    names = []
    si_sdr = []
    pesq = []
    stoi = []

    for key, display_name in model_order:
        if key not in data:
            continue
        names.append(display_name)
        si_sdr.append(data[key].get('si_sdr_db', 0))
        pesq.append(data[key].get('pesq_score', 0))
        stoi.append(data[key].get('stoi_score', 0))

    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 颜色方案
    trad_color = '#E74C3C'
    dl_colors = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    colors = [trad_color] * 3 + dl_colors[:len(names) - 3]

    # SI-SDR
    bars1 = axes[0].bar(x, si_sdr, width * 3, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_title('SI-SDR (dB)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('SI-SDR (dB)', fontsize=11)
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars1, si_sdr):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # PESQ
    bars2 = axes[1].bar(x, pesq, width * 3, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_title('PESQ', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('PESQ Score', fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars2, pesq):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # STOI
    bars3 = axes[2].bar(x, stoi, width * 3, color=colors, edgecolor='white', linewidth=0.5)
    axes[2].set_title('STOI', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('STOI Score', fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars3, stoi):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = output_dir / 'figure_performance_comparison.png'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"性能对比柱状图已保存: {save_path}")


# ============ D6: 训练过程曲线图 ============
def generate_training_curves(output_dir, dpi=300):
    """生成图4-X 训练过程曲线（消融实验的训练/验证损失）"""
    ablation_path = PROJECT_ROOT / 'experiments/exp3_ablation/ablation_results.json'
    if not ablation_path.exists():
        print(f"跳过训练曲线: {ablation_path} 不存在")
        return

    data = load_json(ablation_path)

    config_display = {
        'baseline': 'Baseline',
        '+attention': '+Attention',
        '+residual': '+Residual',
        '+dilated': '+Dilated Conv',
        '+attn+res': '+Attn+Res',
        '+attn+dilated': '+Attn+Dil',
        '+res+dilated': '+Res+Dil',
        'full(+attn+res+dil)': 'Full (Optimized)',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 选择关键配置绘制（避免太密集）
    key_configs = ['baseline', '+residual', '+attention', '+attn+res', 'full(+attn+res+dil)']
    cmap = plt.cm.Set2(np.linspace(0, 1, len(key_configs)))

    for i, cfg_name in enumerate(key_configs):
        cfg_data = None
        for item in data:
            if item['config'] == cfg_name:
                cfg_data = item
                break
        if cfg_data is None:
            continue

        label = config_display.get(cfg_name, cfg_name)
        epochs = range(1, len(cfg_data['train_losses']) + 1)

        ax1.plot(epochs, cfg_data['train_losses'], color=cmap[i], label=label, linewidth=1.5, alpha=0.9)
        ax2.plot(epochs, cfg_data['val_losses'], color=cmap[i], label=label, linewidth=1.5, alpha=0.9)

    ax1.set_title('Training Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_title('Validation Loss', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'figure_training_curves.png'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"训练曲线图已保存: {save_path}")


# ============ 消融实验热力图 ============
def generate_ablation_heatmap(output_dir, dpi=300):
    """生成消融实验结果热力图"""
    ablation_path = PROJECT_ROOT / 'experiments/exp3_ablation/ablation_results.json'
    if not ablation_path.exists():
        print(f"跳过消融热力图: {ablation_path} 不存在")
        return

    data = load_json(ablation_path)

    metrics = ['si_sdr_db', 'pesq_score', 'stoi_score']
    metric_labels = ['SI-SDR (dB)', 'PESQ', 'STOI']

    config_display = {
        'baseline': 'Baseline',
        '+attention': '+Attention',
        '+residual': '+Residual',
        '+dilated': '+Dilated Conv',
        '+attn+res': '+Attn+Res',
        '+attn+dilated': '+Attn+Dil',
        '+res+dilated': '+Res+Dil',
        'full(+attn+res+dil)': 'Full (Optimized)',
    }

    configs = [item['config'] for item in data]
    display_names = [config_display.get(c, c) for c in configs]

    values = np.array([[item[m] for item in data] for m in metrics])

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=11)

    # 添加数值标注
    for i in range(len(metric_labels)):
        for j in range(len(display_names)):
            val = values[i, j]
            text_color = 'white' if val > np.median(values[i]) else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    ax.set_title('消融实验结果热力图', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    save_path = output_dir / 'figure_ablation_heatmap.png'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"消融实验热力图已保存: {save_path}")


# ============ 损失函数对比柱状图 ============
def generate_loss_comparison(output_dir, dpi=300):
    """生成损失函数对比柱状图"""
    loss_path = PROJECT_ROOT / 'experiments/exp4_loss/training_results.json'
    if not loss_path.exists():
        print(f"跳过损失函数对比: {loss_path} 不存在")
        return

    data = load_json(loss_path)

    loss_names = []
    si_sdr = []
    pesq = []
    stoi = []

    loss_display = {
        'mse': 'MSE',
        'l1': 'L1',
        'multi_resolution_stft': 'Multi-Res STFT',
        'spectral_convergence': 'Spectral Conv.',
        'composite': 'Composite',
    }

    for item in data:
        name = item.get('loss_name', item.get('config', 'unknown'))
        loss_names.append(loss_display.get(name, name))
        si_sdr.append(item.get('si_sdr_db', 0))
        pesq.append(item.get('pesq_score', 0))
        stoi.append(item.get('stoi_score', 0))

    x = np.arange(len(loss_names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']

    # SI-SDR
    bars = axes[0].bar(x, si_sdr, width * 3, color=colors[:len(x)], edgecolor='white')
    axes[0].set_title('SI-SDR (dB)', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(loss_names, rotation=25, ha='right', fontsize=9)
    for bar, val in zip(bars, si_sdr):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # PESQ
    bars = axes[1].bar(x, pesq, width * 3, color=colors[:len(x)], edgecolor='white')
    axes[1].set_title('PESQ', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(loss_names, rotation=25, ha='right', fontsize=9)
    for bar, val in zip(bars, pesq):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # STOI
    bars = axes[2].bar(x, stoi, width * 3, color=colors[:len(x)], edgecolor='white')
    axes[2].set_title('STOI', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(loss_names, rotation=25, ha='right', fontsize=9)
    for bar, val in zip(bars, stoi):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = output_dir / 'figure_loss_comparison.png'
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"损失函数对比图已保存: {save_path}")


# ============ 主流程 ============
def main():
    parser = argparse.ArgumentParser(description='生成论文额外图片')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/thesis_figures',
                        help='输出目录')
    parser.add_argument('--dpi', type=int, default=300, help='图片分辨率')
    args = parser.parse_args()

    setup_chinese_font()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("生成论文图片")
    print("=" * 50)

    # D5: 性能演进柱状图
    print("\n[D5] 生成模型性能演进柱状图...")
    generate_performance_bar_chart(output_dir, args.dpi)

    # D6: 训练过程曲线
    print("\n[D6] 生成训练过程曲线图...")
    generate_training_curves(output_dir, args.dpi)

    # 消融实验热力图
    print("\n[补充] 生成消融实验热力图...")
    generate_ablation_heatmap(output_dir, args.dpi)

    # 损失函数对比图
    print("\n[补充] 生成损失函数对比图...")
    generate_loss_comparison(output_dir, args.dpi)

    print("\n" + "=" * 50)
    print(f"全部图片已保存到: {output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
