"""实验1: 统一训练所有U-Net模型变体

按统一设置训练所有模型变体，确保公平对比:
- unet_v1 (3层基线)
- unet_v2 (5层基线)
- unet_v3_attention (5层+注意力)
- unet_v6_optimized (5层+注意力+残差+空洞)
- unet_v10_gan (5层+GAN)

用法:
    python scripts/train_all_models.py
    python scripts/train_all_models.py --epochs 100 --batch-size 8
    python scripts/train_all_models.py --models unet_v2 unet_v6_optimized
    python scripts/train_all_models.py --debug
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 所有模型变体及其配置文件
ALL_MODELS = {
    'unet_v1': 'configs/unet_v1.yaml',
    'unet_v2': 'configs/unet_v2.yaml',
    'unet_v3_attention': 'configs/unet_v3_attention.yaml',
    'unet_v6_optimized': 'configs/unet_v6_optimized.yaml',
    'unet_v10_gan': 'configs/unet_v10_gan.yaml',
}

MODEL_DESCRIPTIONS = {
    'unet_v1': '3层轻量级U-Net基线',
    'unet_v2': '5层标准U-Net基线',
    'unet_v3_attention': '5层U-Net + 注意力门',
    'unet_v6_optimized': '5层U-Net + 注意力+残差+空洞',
    'unet_v10_gan': '5层U-Net + GAN框架',
}


def parse_args():
    parser = argparse.ArgumentParser(description='实验1: 统一训练所有U-Net模型变体')
    parser.add_argument(
        '--models', nargs='+', default=None,
        choices=list(ALL_MODELS.keys()),
        help='指定要训练的模型（默认全部）'
    )
    parser.add_argument('--epochs', type=int, default=None, help='统一训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='统一批大小')
    parser.add_argument('--lr', type=float, default=None, help='统一学习率')
    parser.add_argument('--loss', type=str, default=None,
                        choices=['l1', 'mse', 'spectral', 'multitask', 'multitask_consistency', 'adversarial'],
                        help='统一损失函数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='experiments/exp1_train_all',
                        help='实验结果输出目录')
    parser.add_argument('--debug', action='store_true', help='调试模式（3 epochs）')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已有检查点的模型')
    return parser.parse_args()


def find_checkpoint(model_name):
    """在实验目录中查找已存在的最佳检查点"""
    exp_dir = PROJECT_ROOT / 'experiments'
    if not exp_dir.exists():
        return None
    for exp_path in sorted(exp_dir.iterdir(), reverse=True):
        ckpt = exp_path / 'checkpoints' / 'best_model.pth'
        if exp_path.is_dir() and ckpt.exists():
            # 检查是否是目标模型
            config_path = exp_path / 'config.json'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                if model_name in cfg.get('model', '').lower() or \
                   model_name.replace('unet_', 'AudioUNet') in cfg.get('model', ''):
                    return str(ckpt)
    return None


def train_single_model(model_name, config_path, args):
    """训练单个模型"""
    print(f"\n{'='*70}")
    print(f"  训练模型: {model_name} - {MODEL_DESCRIPTIONS[model_name]}")
    print(f"  配置文件: {config_path}")
    print(f"{'='*70}\n")

    cmd = [sys.executable, '-m', 'src.train', '--config', config_path]

    # 统一参数覆盖
    if args.epochs is not None:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(['--batch-size', str(args.batch_size)])
    if args.lr is not None:
        cmd.extend(['--lr', str(args.lr)])
    if args.loss is not None:
        cmd.extend(['--loss', args.loss])
    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed)])
    if args.debug:
        cmd.append('--debug')

    cmd.extend(['--exp-name', f'unified_{model_name}'])

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)

    print(f"  命令: {' '.join(cmd)}\n")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=False, text=True,
            env=env,
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0

        if success:
            print(f"\n  {model_name} 训练完成 ({elapsed:.1f}s)")
        else:
            print(f"\n  {model_name} 训练失败 (返回码: {result.returncode})")

        return {
            'model': model_name,
            'success': success,
            'elapsed_seconds': elapsed,
            'return_code': result.returncode,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  {model_name} 训练异常: {e}")
        return {
            'model': model_name,
            'success': False,
            'elapsed_seconds': elapsed,
            'error': str(e),
        }


def main():
    args = parse_args()
    models_to_train = args.models or list(ALL_MODELS.keys())

    print("=" * 70)
    print("  实验1: 统一训练所有U-Net模型变体")
    print("=" * 70)
    print(f"\n  待训练模型 ({len(models_to_train)}个):")
    for m in models_to_train:
        print(f"    - {m}: {MODEL_DESCRIPTIONS[m]}")

    if args.epochs:
        print(f"\n  统一训练轮数: {args.epochs}")
    if args.batch_size:
        print(f"  统一批大小: {args.batch_size}")
    if args.lr:
        print(f"  统一学习率: {args.lr}")
    if args.loss:
        print(f"  统一损失函数: {args.loss}")
    print(f"  随机种子: {args.seed}")

    # 创建输出目录
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 依次训练每个模型
    results = []
    total_start = time.time()

    for model_name in models_to_train:
        config_path = ALL_MODELS[model_name]

        if not os.path.exists(str(PROJECT_ROOT / config_path)):
            print(f"\n  跳过 {model_name}: 配置文件不存在 ({config_path})")
            results.append({
                'model': model_name,
                'success': False,
                'error': f'配置文件不存在: {config_path}',
            })
            continue

        if args.skip_existing:
            existing_ckpt = find_checkpoint(model_name)
            if existing_ckpt:
                print(f"\n  跳过 {model_name}: 已有检查点 ({existing_ckpt})")
                results.append({
                    'model': model_name,
                    'success': True,
                    'skipped': True,
                    'checkpoint': existing_ckpt,
                })
                continue

        result = train_single_model(model_name, config_path, args)
        results.append(result)

    total_elapsed = time.time() - total_start

    # 汇总报告
    summary = {
        'experiment': 'exp1_train_all_models',
        'total_models': len(models_to_train),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'total_time_seconds': total_elapsed,
        'unified_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'loss': args.loss,
            'seed': args.seed,
        },
        'results': results,
    }

    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False, default=str)

    # 打印摘要
    print(f"\n{'='*70}")
    print("  训练总结")
    print(f"{'='*70}")
    print(f"  总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  成功: {summary['successful']}/{summary['total_models']}")
    for r in results:
        status = 'OK' if r['success'] else 'FAIL'
        elapsed = f"{r.get('elapsed_seconds', 0):.1f}s"
        print(f"    [{status}] {r['model']:20s} {elapsed}")
    print(f"\n  报告已保存: {summary_path}")


if __name__ == '__main__':
    main()
