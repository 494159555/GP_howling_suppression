"""实验3（改进版）: 消融实验 - 多随机种子 + 多GPU并行

基于AudioUNet5Optimized架构进行消融实验，分析各组件贡献：
- 8种消融配置 × 3个随机种子 = 24个独立训练任务
- 支持多GPU并行：将24个任务均匀分配到可用GPU上

用法:
    # 单GPU
    python scripts/exp3_ablation_study_v2.py

    # 多GPU并行（推荐，8卡可并行8个任务）
    python scripts/exp3_ablation_study_v2.py --gpus 0 1 2 3 4 5 6 7

    # 指定GPU数量（自动选择前N张卡）
    python scripts/exp3_ablation_study_v2.py --num-gpus 4

    # 只跑部分配置
    python scripts/exp3_ablation_study_v2.py --gpus 0 1 --components baseline full

    # 调试
    python scripts/exp3_ablation_study_v2.py --debug --gpus 0
"""

import argparse
import copy
import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.dataset import HowlingDataset
from src.models.blocks import make_encoder_block, make_decoder_block, make_output_block
from src.models.modules.attention_modules import AttentionBlock, ResidualBlock, AtrousConvBlock
from src.models.loss_functions import SpectralLoss


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

        if use_residual:
            self.res1 = ResidualBlock(16)
            self.res2 = ResidualBlock(32)
            self.res3 = ResidualBlock(64)
            self.res4 = ResidualBlock(128)
            self.res5 = ResidualBlock(256)

        if use_dilated:
            self.atrous_block = AtrousConvBlock(256, 256, dilation_rates=self.dilation_rates)

        self.dec5 = make_decoder_block(256, 128)
        self.dec4 = make_decoder_block(256, 64)
        self.dec3 = make_decoder_block(128, 32)
        self.dec2 = make_decoder_block(64, 16)
        self.dec1 = make_output_block(32)

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
    ('baseline',              False, False, False),
    ('+attention',            True,  False, False),
    ('+residual',             False, True,  False),
    ('+dilated',              False, False, True),
    ('+attn+res',             True,  True,  False),
    ('+attn+dilated',         True,  False, True),
    ('+res+dilated',          False, True,  True),
    ('full(+attn+res+dil)',   True,  True,  True),
]


def set_seed(seed):
    """设置所有随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def worker_fn(gpu_id, task_queue, result_dict, args_kwargs):
    """工作进程函数：从队列取任务，在指定GPU上训练"""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))

    device = torch.device(f'cuda:{gpu_id}')
    from src.evaluation.metrics import AudioMetrics
    from src.evaluate import _istft_from_mag_phase

    # 每个worker独立加载数据集（避免共享DataLoader的多进程问题）
    train_dataset = HowlingDataset(cfg.TRAIN_CLEAN_DIR, cfg.TRAIN_NOISY_DIR)
    val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
    val_dataset_td = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR, return_waveform=True)
    train_loader = DataLoader(train_dataset, batch_size=args_kwargs['batch_size'],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args_kwargs['batch_size'],
                            shuffle=False, num_workers=2, pin_memory=True)
    val_loader_td = DataLoader(val_dataset_td, batch_size=args_kwargs['batch_size'],
                               shuffle=False, num_workers=2, pin_memory=True)
    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)

    while True:
        task = task_queue.get()
        if task is None:  # 毒丸信号，退出
            break

        task_id, config_name, use_attention, use_residual, use_dilated, seed = task

        try:
            set_seed(seed)
            model = AblationUNet(
                use_attention=use_attention,
                use_residual=use_residual,
                use_dilated=use_dilated,
            ).to(device)

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            criterion = SpectralLoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=1e-3, weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

            best_val_loss = float('inf')
            best_model_state = None
            epochs = args_kwargs['epochs']

            for epoch in range(epochs):
                model.train()
                for noisy, clean in train_loader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    optimizer.zero_grad()
                    pred = model(noisy)
                    loss = criterion(pred, clean)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for noisy, clean in val_loader:
                        noisy, clean = noisy.to(device), clean.to(device)
                        pred = model(noisy)
                        val_loss += criterion(pred, clean).item()
                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())

                if (epoch + 1) % 25 == 0 or epoch == 0:
                    print(f"  [GPU {gpu_id}] {config_name} seed={seed} "
                          f"Epoch [{epoch+1}/{epochs}] Val: {val_loss:.4f}")

            # 加载最佳模型并评估
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            model.eval()

            all_si_sdr = []
            all_pesq = []
            all_stoi = []

            with torch.no_grad():
                td_iter = iter(val_loader_td)
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    pred = model(noisy)
                    try:
                        td_batch = next(td_iter)
                        td_noisy, td_clean, td_noisy_wave, td_clean_wave, td_noisy_stft = td_batch
                        td_noisy = td_noisy.to(device)
                        td_pred = model(td_noisy)
                        enhanced_wave = _istft_from_mag_phase(
                            td_pred.cpu(), td_noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH
                        )
                        all_si_sdr.append(metrics_calc.calculate_si_sdr(td_clean_wave, enhanced_wave))
                        all_stoi.append(metrics_calc.calculate_stoi(td_clean_wave, enhanced_wave))
                        all_pesq.append(metrics_calc.calculate_pesq(td_clean_wave, enhanced_wave))
                    except StopIteration:
                        break
                    except Exception:
                        pass

            result = {
                'task_id': task_id,
                'config': config_name,
                'seed': seed,
                'use_attention': use_attention,
                'use_residual': use_residual,
                'use_dilated': use_dilated,
                'param_count': param_count,
                'best_val_loss': best_val_loss,
                'si_sdr_db': float(np.mean(all_si_sdr)) if all_si_sdr else 0.0,
                'pesq_score': float(np.mean(all_pesq)) if all_pesq else 0.0,
                'stoi_score': float(np.mean(all_stoi)) if all_stoi else 0.0,
            }

            print(f"  [GPU {gpu_id}] DONE {config_name} seed={seed} "
                  f"SI-SDR={result['si_sdr_db']:.2f} PESQ={result['pesq_score']:.2f} "
                  f"STOI={result['stoi_score']:.4f}")
            result_dict[task_id] = result

        except Exception as e:
            print(f"  [GPU {gpu_id}] FAIL {config_name} seed={seed}: {e}")
            import traceback
            traceback.print_exc()
            result_dict[task_id] = {
                'task_id': task_id,
                'config': config_name,
                'seed': seed,
                'error': str(e),
            }


def main():
    parser = argparse.ArgumentParser(description='实验3（改进版）: 消融实验（多种子 + 多GPU）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16, help='批大小')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='随机种子列表')
    parser.add_argument('--output-dir', type=str, default='experiments/exp3_ablation_v2',
                        help='输出目录')
    parser.add_argument('--debug', action='store_true', help='调试模式（5 epochs, 1 seed）')
    parser.add_argument('--components', nargs='+', default=None,
                        choices=['baseline', '+attention', '+residual', '+dilated',
                                 '+attn+res', '+attn+dilated', '+res+dilated', 'full'],
                        help='只运行指定配置的消融实验')
    parser.add_argument('--gpus', nargs='+', type=int, default=None,
                        help='指定使用的GPU ID列表，如 --gpus 0 1 2 3')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='使用前N张GPU（不指定则用所有可用GPU）')
    args = parser.parse_args()

    if args.debug:
        args.epochs = 5
        args.seeds = [42]

    # 确定使用的GPU
    num_available = torch.cuda.device_count()
    if args.gpus is not None:
        gpu_ids = args.gpus
    elif args.num_gpus is not None:
        gpu_ids = list(range(min(args.num_gpus, num_available)))
    else:
        gpu_ids = list(range(num_available))

    num_gpus = len(gpu_ids)
    print(f"可用GPU: {num_available}, 使用GPU: {gpu_ids}")
    print(f"随机种子: {args.seeds}")
    print(f"训练轮数: {args.epochs}")

    # 筛选配置
    configs = ABLATION_CONFIGS
    if args.components:
        configs = [c for c in configs if c[0] in args.components]

    # 生成任务列表
    tasks = []
    task_id = 0
    for name, attn, res, dil in configs:
        for seed in args.seeds:
            tasks.append((task_id, name, attn, res, dil, seed))
            task_id += 1

    total_tasks = len(tasks)
    print(f"\n消融配置 ({len(configs)}个) × 种子 ({len(args.seeds)}个) = {total_tasks} 个任务")
    print(f"并行度: {num_gpus} GPU, 预计 {((total_tasks + num_gpus - 1) // num_gpus)} 轮")

    for name, attn, res, dil in configs:
        flags = []
        if attn: flags.append('注意力')
        if res: flags.append('残差')
        if dil: flags.append('空洞')
        print(f"  {name:25s} [{', '.join(flags) or '无'}]")

    # 输出目录
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    args_kwargs = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
    }

    if num_gpus <= 1 or total_tasks <= 1:
        # ====== 单GPU模式（兼容原逻辑）======
        print(f"\n使用单GPU模式: cuda:{gpu_ids[0]}")
        device = torch.device(f'cuda:{gpu_ids[0]}')

        print("\n加载数据集...")
        train_dataset = HowlingDataset(cfg.TRAIN_CLEAN_DIR, cfg.TRAIN_NOISY_DIR)
        val_dataset = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR)
        val_dataset_td = HowlingDataset(cfg.VAL_CLEAN_DIR, cfg.VAL_NOISY_DIR, return_waveform=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
        val_loader_td = DataLoader(val_dataset_td, batch_size=args.batch_size, shuffle=False,
                                   num_workers=4, pin_memory=True)
        print(f"训练: {len(train_dataset)}, 验证: {len(val_dataset)}")

        from src.evaluation.metrics import AudioMetrics
        from src.evaluate import _istft_from_mag_phase
        metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)

        all_results = []
        for run_idx, (tid, name, attn, res, dil, seed) in enumerate(tasks):
            print(f"\n{'='*60}")
            print(f"  [{run_idx+1}/{total_tasks}] {name} (seed={seed})")
            print(f"{'='*60}")

            set_seed(seed)
            model = AblationUNet(use_attention=attn, use_residual=res, use_dilated=dil).to(device)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            criterion = SpectralLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5)

            best_val_loss = float('inf')
            best_model_state = None

            for epoch in range(args.epochs):
                model.train()
                for noisy, clean in train_loader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    optimizer.zero_grad()
                    pred = model(noisy)
                    loss = criterion(pred, clean)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for noisy, clean in val_loader:
                        noisy, clean = noisy.to(device), clean.to(device)
                        val_loss += criterion(model(noisy), clean).item()
                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())

                if (epoch + 1) % 25 == 0 or epoch == 0:
                    print(f"    Epoch [{epoch+1}/{args.epochs}] Val: {val_loss:.4f}")

            if best_model_state:
                model.load_state_dict(best_model_state)
            model.eval()

            all_si_sdr, all_pesq, all_stoi = [], [], []
            with torch.no_grad():
                td_iter = iter(val_loader_td)
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    pred = model(noisy)
                    try:
                        td_batch = next(td_iter)
                        td_noisy, td_clean, td_noisy_wave, td_clean_wave, td_noisy_stft = td_batch
                        td_noisy = td_noisy.to(device)
                        td_pred = model(td_noisy)
                        ew = _istft_from_mag_phase(td_pred.cpu(), td_noisy_stft, cfg.N_FFT, cfg.HOP_LENGTH)
                        all_si_sdr.append(metrics_calc.calculate_si_sdr(td_clean_wave, ew))
                        all_stoi.append(metrics_calc.calculate_stoi(td_clean_wave, ew))
                        all_pesq.append(metrics_calc.calculate_pesq(td_clean_wave, ew))
                    except StopIteration:
                        break
                    except Exception:
                        pass

            result = {
                'config': name, 'seed': seed,
                'use_attention': attn, 'use_residual': res, 'use_dilated': dil,
                'param_count': param_count, 'best_val_loss': best_val_loss,
                'si_sdr_db': float(np.mean(all_si_sdr)) if all_si_sdr else 0.0,
                'pesq_score': float(np.mean(all_pesq)) if all_pesq else 0.0,
                'stoi_score': float(np.mean(all_stoi)) if all_stoi else 0.0,
            }
            all_results.append(result)
            print(f"  SI-SDR: {result['si_sdr_db']:.2f}, PESQ: {result['pesq_score']:.2f}, "
                  f"STOI: {result['stoi_score']:.4f}")

    else:
        # ====== 多GPU并行模式 ======
        print(f"\n使用多GPU并行模式: {num_gpus} GPU")
        print("启动worker进程...")

        mp.set_start_method('spawn', force=True)
        manager = mp.Manager()
        task_queue = manager.Queue()
        result_dict = manager.dict()

        # 将任务放入队列
        for task in tasks:
            task_queue.put(task)

        # 放入毒丸信号（每个worker一个）
        for _ in range(num_gpus):
            task_queue.put(None)

        # 启动worker进程
        workers = []
        for gpu_id in gpu_ids:
            p = mp.Process(
                target=worker_fn,
                args=(gpu_id, task_queue, result_dict, args_kwargs),
            )
            p.start()
            workers.append(p)
            print(f"  Worker started: GPU {gpu_id} (PID {p.pid})")

        # 等待所有worker完成
        for p in workers:
            p.join()

        print(f"\n所有worker已完成，收集结果...")
        all_results = []
        for tid in sorted(result_dict.keys()):
            r = dict(result_dict[tid])
            if 'error' not in r:
                all_results.append(r)
            else:
                print(f"  任务 {tid} ({r.get('config', '?')} seed={r.get('seed', '?')}) 失败: {r['error']}")

    # ====== 统计分析 ======
    print(f"\n{'='*100}")
    print("  消融实验结果（多随机种子统计）")
    print(f"{'='*100}")
    header = (f"{'配置':25s} | {'参数量':>10s} | "
              f"{'SI-SDR (dB)':>20s} | {'PESQ':>20s} | {'STOI':>20s}")
    print(header)
    print("-" * 100)

    summary_results = []
    for name, attn, res, dil in configs:
        config_results = [r for r in all_results if r['config'] == name]
        if not config_results:
            continue

        si_sdrs = [r['si_sdr_db'] for r in config_results]
        pesqs = [r['pesq_score'] for r in config_results]
        stois = [r['stoi_score'] for r in config_results]
        param_count = config_results[0]['param_count']

        summary = {
            'config': name,
            'use_attention': attn,
            'use_residual': res,
            'use_dilated': dil,
            'param_count': param_count,
            'num_seeds': len(config_results),
            'si_sdr_mean': float(np.mean(si_sdrs)),
            'si_sdr_std': float(np.std(si_sdrs)),
            'pesq_mean': float(np.mean(pesqs)),
            'pesq_std': float(np.std(pesqs)),
            'stoi_mean': float(np.mean(stois)),
            'stoi_std': float(np.std(stois)),
            'seed_results': config_results,
        }
        summary_results.append(summary)

        print(f"{name:25s} | {param_count:>10,d} | "
              f"{summary['si_sdr_mean']:8.2f} ± {summary['si_sdr_std']:.2f} | "
              f"{summary['pesq_mean']:8.2f} ± {summary['pesq_std']:.2f} | "
              f"{summary['stoi_mean']:8.4f} ± {summary['stoi_std']:.4f}")

    # 边际贡献分析
    print(f"\n{'='*60}")
    print("  边际贡献分析")
    print(f"{'='*60}")

    baseline_s = next((s for s in summary_results if s['config'] == 'baseline'), None)
    full_s = next((s for s in summary_results if s['config'] == 'full(+attn+res+dil)'), None)

    if baseline_s and full_s:
        print(f"\n  Baseline SI-SDR: {baseline_s['si_sdr_mean']:.2f} ± {baseline_s['si_sdr_std']:.2f} dB")
        print(f"  Full     SI-SDR: {full_s['si_sdr_mean']:.2f} ± {full_s['si_sdr_std']:.2f} dB")
        print(f"  Full vs Baseline: {full_s['si_sdr_mean'] - baseline_s['si_sdr_mean']:+.2f} dB")

    for target_name, label in [
        ('+attention', '注意力（单独）'),
        ('+residual', '残差连接（单独）'),
        ('+dilated', '空洞卷积（单独）'),
    ]:
        target = next((s for s in summary_results if s['config'] == target_name), None)
        if baseline_s and target:
            delta = target['si_sdr_mean'] - baseline_s['si_sdr_mean']
            print(f"  {label}: {delta:+.2f} dB (vs Baseline)")

    # 保存
    results_path = output_dir / 'ablation_results_v2.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'all_results': all_results,
            'summary': summary_results,
            'config': {
                'seeds': args.seeds,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'gpus': gpu_ids,
            }
        }, f, indent=4, ensure_ascii=False, default=str)

    print(f"\n结果已保存: {results_path}")


if __name__ == '__main__':
    main()
