"""实验6: 生成对比表格(CSV)和可视化图表

从所有实验结果JSON文件中读取数据，生成:
1. 对比表格 (CSV)
2. 柱状图/条形图
3. 雷达图
4. 训练曲线对比图
5. 消融实验热力图
6. 综合报告

用法:
    python scripts/generate_report.py
    python scripts/generate_report.py --exp-dir experiments
    python scripts/generate_report.py --exp2 experiments/exp2_eval_all
    python scripts/generate_report.py --format pdf
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

PROJECT_ROOT = Path(__file__).parent.parent

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ========== 工具函数 ==========

def load_json(path):
    """加载JSON文件"""
    if not os.path.exists(path):
        print(f"  警告: 文件不存在 - {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def safe_float(val, default=0.0):
    """安全转换为float"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ========== CSV 生成 ==========

def generate_eval_csv(data, output_path):
    """生成模型评估对比CSV"""
    rows = []
    for name, metrics in data.items():
        rows.append({
            '方法': name,
            'L1 Loss': safe_float(metrics.get('avg_loss')),
            'SNR改善(dB)': safe_float(metrics.get('snr_improvement_db')),
            'PSNR(dB)': safe_float(metrics.get('psnr_db')),
            'STOI': safe_float(metrics.get('stoi_score')),
            '啸叫抑制(dB)': safe_float(metrics.get('howling_reduction_db')),
            '频谱平滑度改善': safe_float(metrics.get('spectral_smoothness_improvement')),
            '高频衰减': safe_float(metrics.get('high_frequency_reduction')),
            'MOS估算': safe_float(metrics.get('mos_estimate')),
            '推理时间(ms)': safe_float(metrics.get('avg_inference_ms')),
            '参数量': safe_float(metrics.get('param_count'), 0),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('MOS估算', ascending=False)
    csv_path = output_path / 'evaluation_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV已保存: {csv_path}")
    return df


def generate_ablation_csv(data, output_path):
    """生成消融实验CSV"""
    rows = []
    for r in data:
        rows.append({
            '配置': r.get('config', ''),
            '注意力': 'Y' if r.get('use_attention') else 'N',
            '残差': 'Y' if r.get('use_residual') else 'N',
            '空洞卷积': 'Y' if r.get('use_dilated') else 'N',
            '参数量': safe_float(r.get('param_count'), 0),
            'L1 Loss': safe_float(r.get('avg_loss')),
            'Best Val Loss': safe_float(r.get('best_val_loss')),
            'SNR改善(dB)': safe_float(r.get('snr_improvement_db')),
            'PSNR(dB)': safe_float(r.get('psnr_db')),
            'STOI': safe_float(r.get('stoi_score')),
            '啸叫抑制(dB)': safe_float(r.get('howling_reduction_db')),
            'MOS估算': safe_float(r.get('mos_estimate')),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('MOS估算', ascending=False)
    csv_path = output_path / 'ablation_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV已保存: {csv_path}")
    return df


def generate_loss_csv(data, output_path):
    """生成损失函数对比CSV"""
    rows = []
    for r in data:
        rows.append({
            '损失函数': r.get('loss_name', ''),
            '类型': r.get('loss_type', ''),
            'Best Val Loss': safe_float(r.get('best_val_loss')),
            '统一L1评估': safe_float(r.get('avg_l1_loss')),
            'SNR改善(dB)': safe_float(r.get('snr_improvement_db')),
            'PSNR(dB)': safe_float(r.get('psnr_db')),
            'STOI': safe_float(r.get('stoi_score')),
            'MOS估算': safe_float(r.get('mos_estimate')),
            '训练耗时(s)': safe_float(r.get('training_time_seconds')),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('MOS估算', ascending=False)
    csv_path = output_path / 'loss_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV已保存: {csv_path}")
    return df


def generate_strategy_csv(data, output_path):
    """生成训练策略对比CSV"""
    rows = []
    for r in data:
        params = r.get('strategy_params', {})
        rows.append({
            '策略': r.get('strategy_name', ''),
            'LR调度器': params.get('lr_scheduler', ''),
            '混合精度': 'Y' if params.get('mixed_precision') else 'N',
            'Warmup轮数': params.get('warmup_epochs', 0),
            'Best Val Loss': safe_float(r.get('best_val_loss')),
            '统一L1评估': safe_float(r.get('avg_l1_loss')),
            'SNR改善(dB)': safe_float(r.get('snr_improvement_db')),
            'MOS估算': safe_float(r.get('mos_estimate')),
            '训练耗时(s)': safe_float(r.get('training_time_seconds')),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('MOS估算', ascending=False)
    csv_path = output_path / 'strategy_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV已保存: {csv_path}")
    return df


def generate_augmentation_csv(data, output_path):
    """生成数据增强对比CSV"""
    rows = []
    for r in data:
        rows.append({
            '数据增强': r.get('augmentation', ''),
            'Best Val Loss': safe_float(r.get('best_val_loss')),
            '统一L1评估': safe_float(r.get('avg_l1_loss')),
            'SNR改善(dB)': safe_float(r.get('snr_improvement_db')),
            'STOI': safe_float(r.get('stoi_score')),
            'MOS估算': safe_float(r.get('mos_estimate')),
            '训练耗时(s)': safe_float(r.get('training_time_seconds')),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('MOS估算', ascending=False)
    csv_path = output_path / 'augmentation_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV已保存: {csv_path}")
    return df


# ========== 可视化 ==========

def plot_metrics_bar(df, output_path, title='模型评估对比'):
    """柱状图: 关键指标对比"""
    metrics_cols = ['SNR改善(dB)', 'PSNR(dB)', 'STOI', '啸叫抑制(dB)', 'MOS估算']
    available_cols = [c for c in metrics_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(available_cols), figsize=(5 * len(available_cols), 6))
    if len(available_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, available_cols):
        values = df[col].values
        names = df['方法'].values if '方法' in df.columns else df.index
        colors = plt.cm.Set2(np.linspace(0, 1, len(values)))

        bars = ax.barh(range(len(values)), values, color=colors)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel(col)
        ax.set_title(col)
        ax.invert_yaxis()

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01 * max(abs(values)), bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_path / 'metrics_comparison_bar.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {path}")


def plot_radar(data, output_path, title='多维度对比雷达图'):
    """雷达图: 多维度对比"""
    categories = ['SNR改善', 'PSNR', 'STOI', '啸叫抑制', 'MOS']
    cat_keys = ['snr_improvement_db', 'psnr_db', 'stoi_score', 'howling_reduction_db', 'mos_estimate']

    # 归一化
    max_vals = {}
    for key in cat_keys:
        vals = [safe_float(m.get(key, 0)) for m in data.values()]
        max_vals[key] = max(abs(v) for v in vals) if vals else 1
        if max_vals[key] == 0:
            max_vals[key] = 1

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(data)))

    for (name, metrics), color in zip(data.items(), colors):
        values = [safe_float(metrics.get(k, 0)) / max_vals[k] for k in cat_keys]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    path = output_path / 'radar_chart.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {path}")


def plot_training_curves(results_list, output_path, title='训练曲线对比'):
    """训练曲线对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for r in results_list:
        name = r.get('config', r.get('loss_name', r.get('strategy_name',
                r.get('augmentation', r.get('model', 'unknown')))))
        train_losses = r.get('train_losses', [])
        val_losses = r.get('val_losses', [])

        if train_losses:
            ax1.plot(train_losses, label=name)
        if val_losses:
            ax2.plot(val_losses, label=name)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('验证损失')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_path / 'training_curves.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {path}")


def plot_ablation_heatmap(data, output_path):
    """消融实验热力图"""
    configs = []
    metrics = {
        'SNR改善': [], 'STOI': [], '啸叫抑制': [], 'MOS': [],
    }

    for r in data:
        configs.append(r.get('config', ''))
        metrics['SNR改善'].append(safe_float(r.get('snr_improvement_db')))
        metrics['STOI'].append(safe_float(r.get('stoi_score')))
        metrics['啸叫抑制'].append(safe_float(r.get('howling_reduction_db')))
        metrics['MOS'].append(safe_float(r.get('mos_estimate')))

    matrix = np.array([metrics[k] for k in metrics])

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(list(metrics.keys()))

    for i in range(len(metrics)):
        for j in range(len(configs)):
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', fontsize=9)

    plt.colorbar(im)
    plt.title('消融实验指标热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_path / 'ablation_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {path}")


def plot_component_contribution(data, output_path):
    """组件贡献分析图"""
    # 基线值
    baseline = None
    for r in data:
        if not r.get('use_attention') and not r.get('use_residual') and not r.get('use_dilated'):
            baseline = r
            break

    if baseline is None:
        print("  跳过组件贡献图: 未找到基线配置")
        return

    # 计算各组件的MOS增量贡献
    components = {
        '注意力': None, '残差': None, '空洞': None,
        '注意力+残差': None, '注意力+空洞': None, '残差+空洞': None,
        '全部': None,
    }

    target_map = {
        '注意力': (True, False, False),
        '残差': (False, True, False),
        '空洞': (False, False, True),
        '注意力+残差': (True, True, False),
        '注意力+空洞': (True, False, True),
        '残差+空洞': (False, True, True),
        '全部': (True, True, True),
    }

    base_mos = safe_float(baseline.get('mos_estimate', 0))

    contributions = {}
    for comp_name, (attn, res, dil) in target_map.items():
        for r in data:
            if (r.get('use_attention') == attn and r.get('use_residual') == res
                    and r.get('use_dilated') == dil):
                contributions[comp_name] = safe_float(r.get('mos_estimate', 0)) - base_mos
                break

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(contributions.keys())
    values = list(contributions.values())
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]

    bars = ax.bar(names, values, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('MOS增量贡献')
    ax.set_title('各组件对MOS的增量贡献', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')

    for bar, val in zip(bars, values):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.01 * (1 if y >= 0 else -1),
                f'{val:+.3f}', ha='center', va='bottom' if y >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    path = output_path / 'component_contribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {path}")


def plot_loss_comparison_bar(data, output_path):
    """损失函数对比柱状图"""
    names = [r.get('loss_name', '') for r in data]
    mos = [safe_float(r.get('mos_estimate', 0)) for r in data]
    snr = [safe_float(r.get('snr_improvement_db', 0)) for r in data]
    time_s = [safe_float(r.get('training_time_seconds', 0)) for r in data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MOS
    axes[0].barh(names, mos, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    axes[0].set_xlabel('MOS')
    axes[0].set_title('MOS估算')
    axes[0].invert_yaxis()

    # SNR
    axes[1].barh(names, snr, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    axes[1].set_xlabel('SNR改善(dB)')
    axes[1].set_title('SNR改善')
    axes[1].invert_yaxis()

    # 训练时间
    axes[2].barh(names, time_s, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    axes[2].set_xlabel('训练时间(s)')
    axes[2].set_title('训练耗时')
    axes[2].invert_yaxis()

    plt.suptitle('损失函数对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_path / 'loss_comparison_bar.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {path}")


def plot_strategy_comparison_bar(data, output_path):
    """训练策略对比柱状图"""
    names = [r.get('strategy_name', '') for r in data]
    mos = [safe_float(r.get('mos_estimate', 0)) for r in data]
    time_s = [safe_float(r.get('training_time_seconds', 0)) for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.barh(names, mos, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
    ax1.set_xlabel('MOS')
    ax1.set_title('MOS估算对比')
    ax1.invert_yaxis()

    ax2.barh(names, time_s, color=plt.cm.Set3(np.linspace(0, 1, len(names))))
    ax2.set_xlabel('训练时间(s)')
    ax2.set_title('训练耗时对比')
    ax2.invert_yaxis()

    plt.suptitle('训练策略对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_path / 'strategy_comparison_bar.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {path}")


def plot_augmentation_comparison_bar(data, output_path):
    """数据增强对比柱状图"""
    names = [r.get('augmentation', '') for r in data]
    mos = [safe_float(r.get('mos_estimate', 0)) for r in data]
    snr = [safe_float(r.get('snr_improvement_db', 0)) for r in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(names, mos, color=plt.cm.Pastel1(np.linspace(0, 1, len(names))))
    ax1.set_ylabel('MOS')
    ax1.set_title('MOS估算对比')
    ax1.tick_params(axis='x', rotation=30)

    ax2.bar(names, snr, color=plt.cm.Pastel1(np.linspace(0, 1, len(names))))
    ax2.set_ylabel('SNR改善(dB)')
    ax2.set_title('SNR改善对比')
    ax2.tick_params(axis='x', rotation=30)

    plt.suptitle('数据增强策略对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = output_path / 'augmentation_comparison_bar.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {path}")


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description='实验7: 生成对比表格和可视化图表')
    parser.add_argument('--exp-dir', type=str, default='experiments',
                        help='实验根目录')
    parser.add_argument('--exp2', type=str, default=None,
                        help='实验2评估结果路径（JSON）')
    parser.add_argument('--exp3', type=str, default=None,
                        help='实验3消融结果路径')
    parser.add_argument('--exp4', type=str, default=None,
                        help='实验4损失对比结果路径')
    parser.add_argument('--exp5', type=str, default=None,
                        help='实验5策略对比结果路径')
    parser.add_argument('--exp6', type=str, default=None,
                        help='实验6增强对比结果路径')
    parser.add_argument('--output-dir', type=str, default='experiments/exp7_report',
                        help='报告输出目录')
    args = parser.parse_args()

    exp_root = PROJECT_ROOT / args.exp_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  实验7: 生成对比表格和可视化图表")
    print("=" * 60)

    # 自动发现实验结果
    exp2_path = args.exp2 or str(exp_root / 'exp2_eval_all' / 'evaluation_results.json')
    exp3_path = args.exp3 or str(exp_root / 'exp3_ablation' / 'ablation_results.json')
    exp4_path = args.exp4 or str(exp_root / 'exp4_loss_comparison' / 'loss_comparison_results.json')
    exp5_path = args.exp5 or str(exp_root / 'exp5_strategy_comparison' / 'strategy_comparison_results.json')
    exp6_path = args.exp6 or str(exp_root / 'exp6_augmentation' / 'augmentation_comparison_results.json')

    # ---- 实验2: 全模型评估 ----
    print("\n[实验2] 全模型评估对比")
    exp2_data = load_json(exp2_path)
    if exp2_data:
        df = generate_eval_csv(exp2_data, output_dir)
        plot_metrics_bar(df, output_dir, title='所有方法评估对比')
        plot_radar(exp2_data, output_dir, title='模型 vs 传统方法 雷达图')

    # ---- 实验3: 消融实验 ----
    print("\n[实验3] 消融实验")
    exp3_data = load_json(exp3_path)
    if exp3_data:
        generate_ablation_csv(exp3_data, output_dir)
        plot_training_curves(exp3_data, output_dir, title='消融实验训练曲线')
        plot_ablation_heatmap(exp3_data, output_dir)
        plot_component_contribution(exp3_data, output_dir)

    # ---- 实验4: 损失函数对比 ----
    print("\n[实验4] 损失函数对比")
    exp4_data = load_json(exp4_path)
    if exp4_data:
        generate_loss_csv(exp4_data, output_dir)
        plot_loss_comparison_bar(exp4_data, output_dir)
        plot_training_curves(exp4_data, output_dir, title='不同损失函数训练曲线')

    # ---- 实验5: 训练策略对比 ----
    print("\n[实验5] 训练策略对比")
    exp5_data = load_json(exp5_path)
    if exp5_data:
        generate_strategy_csv(exp5_data, output_dir)
        plot_strategy_comparison_bar(exp5_data, output_dir)
        plot_training_curves(exp5_data, output_dir, title='不同训练策略训练曲线')

    # ---- 实验6: 数据增强对比 ----
    print("\n[实验6] 数据增强对比")
    exp6_data = load_json(exp6_path)
    if exp6_data:
        generate_augmentation_csv(exp6_data, output_dir)
        plot_augmentation_comparison_bar(exp6_data, output_dir)
        plot_training_curves(exp6_data, output_dir, title='不同数据增强策略训练曲线')

    # ---- 生成总览报告 ----
    report = {
        'experiment_2_eval': exp2_path if exp2_data else '未找到数据',
        'experiment_3_ablation': exp3_path if exp3_data else '未找到数据',
        'experiment_4_loss': exp4_path if exp4_data else '未找到数据',
        'experiment_5_strategy': exp5_path if exp5_data else '未找到数据',
        'experiment_6_augmentation': exp6_path if exp6_data else '未找到数据',
    }
    report_path = output_dir / 'report_summary.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  报告生成完成!")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
