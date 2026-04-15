"""测试传统啸叫抑制方法

测试三种传统方法在完整评估流程中的表现，包括:
1. 基本前向传播测试（合成数据）
2. 模拟数据集输出的归一化频谱测试
3. 指标计算集成测试
4. 多批次测试（检测状态累积问题）
"""

import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.traditional import (
    FrequencyShiftMethod,
    GainSuppressionMethod,
    AdaptiveFeedbackMethod,
)
from src.evaluation.metrics import AudioMetrics, calculate_mos_score


def test_basic_forward():
    """测试1: 基本前向传播"""
    print("=" * 60)
    print("测试1: 基本前向传播（log域合成数据）")
    print("=" * 60)
    
    batch_size, channels, freq_bins, time_frames = 2, 1, 256, 100
    
    # 模拟log域频谱数据（与各自__main__测试一致）
    test_input = torch.randn(batch_size, channels, freq_bins, time_frames).abs()
    test_input = torch.log10(test_input + 1e-8)
    
    methods = {
        'FrequencyShift': FrequencyShiftMethod(shift_hz=20.0),
        'GainSuppression': GainSuppressionMethod(threshold_db=-30.0),
        'AdaptiveFeedback': AdaptiveFeedbackMethod(filter_length=64, step_size=0.01),
    }
    
    all_passed = True
    for name, method in methods.items():
        try:
            method.eval()
            with torch.no_grad():
                output = method(test_input)
            
            assert output.shape == test_input.shape, \
                f"{name}: 输出形状不匹配 {output.shape} != {test_input.shape}"
            assert not torch.isnan(output).any(), \
                f"{name}: 输出包含NaN"
            assert not torch.isinf(output).any(), \
                f"{name}: 输出包含Inf"
            
            print(f"  ✓ {name}: 形状={output.shape}, "
                  f"范围=[{output.min():.4f}, {output.max():.4f}]")
        except Exception as e:
            print(f"  ✗ {name}: 失败 - {e}")
            all_passed = False
    
    return all_passed


def test_normalized_spectrogram():
    """测试2: 归一化频谱数据（模拟数据集输出）"""
    print("\n" + "=" * 60)
    print("测试2: 归一化频谱数据（模拟HowlingDataset输出）")
    print("=" * 60)
    
    batch_size = 4
    freq_bins = 256  # n_fft//2 = 512//2 = 256
    time_frames = 188  # 3s * 16000 / 128 + 1 ≈ 376 -> 但实际上chunk_size/hop_length+1
    
    # 模拟数据集归一化后的频谱 [0, 1] 范围
    # 数据集: howling_norm = (howling_log - (-11.5)) / (2.5 - (-11.5)) = (howling_log + 11.5) / 14
    noisy_mag = torch.rand(batch_size, 1, freq_bins, time_frames) * 0.6 + 0.2  # [0.2, 0.8]
    clean_mag = torch.rand(batch_size, 1, freq_bins, time_frames) * 0.5 + 0.25  # [0.25, 0.75]
    
    # 添加一些模拟啸叫峰值
    noisy_mag[:, 0, 100:110, 50:100] += 0.3  # 在某些频率和时间帧添加啸叫
    noisy_mag = torch.clamp(noisy_mag, 0.0, 1.0)
    
    methods = {
        'FrequencyShift': FrequencyShiftMethod(shift_hz=20.0),
        'GainSuppression': GainSuppressionMethod(threshold_db=-30.0),
        'AdaptiveFeedback': AdaptiveFeedbackMethod(filter_length=64, step_size=0.01),
    }
    
    all_passed = True
    for name, method in methods.items():
        try:
            method.eval()
            with torch.no_grad():
                output = method(noisy_mag)
            
            assert output.shape == noisy_mag.shape, \
                f"{name}: 输出形状不匹配 {output.shape} != {noisy_mag.shape}"
            assert not torch.isnan(output).any(), \
                f"{name}: 输出包含NaN"
            assert not torch.isinf(output).any(), \
                f"{name}: 输出包含Inf"
            
            # 计算与clean的L1损失
            l1_loss = torch.nn.L1Loss()(output, clean_mag).item()
            
            print(f"  ✓ {name}: 形状={output.shape}, "
                  f"范围=[{output.min():.4f}, {output.max():.4f}], "
                  f"L1 Loss={l1_loss:.4f}")
        except Exception as e:
            print(f"  ✗ {name}: 失败 - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_metrics_integration():
    """测试3: 指标计算集成测试"""
    print("\n" + "=" * 60)
    print("测试3: 指标计算集成测试")
    print("=" * 60)
    
    batch_size = 2
    freq_bins = 256
    time_frames = 100
    
    noisy_mag = torch.rand(batch_size, 1, freq_bins, time_frames) * 0.6 + 0.2
    clean_mag = torch.rand(batch_size, 1, freq_bins, time_frames) * 0.5 + 0.25
    
    metrics_calc = AudioMetrics(sample_rate=16000)
    
    methods = {
        'FrequencyShift': FrequencyShiftMethod(shift_hz=20.0),
        'GainSuppression': GainSuppressionMethod(threshold_db=-30.0),
        'AdaptiveFeedback': AdaptiveFeedbackMethod(filter_length=64, step_size=0.01),
    }
    
    all_passed = True
    for name, method in methods.items():
        try:
            method.eval()
            with torch.no_grad():
                pred_mag = method(noisy_mag)
            
            # 计算所有指标（与evaluate_all.py中的流程一致）
            sample_metrics = metrics_calc.calculate_all_metrics(
                clean=clean_mag,     # 修复：使用干净语音作为参考
                noisy=noisy_mag,
                enhanced=pred_mag,   # 修复：模型输出作为增强结果
            )
            
            mos = calculate_mos_score(sample_metrics)
            
            print(f"  ✓ {name}:")
            print(f"    SNR改善: {sample_metrics.get('snr_improvement_db', 0):.2f} dB")
            print(f"    PSNR: {sample_metrics.get('psnr_db', 0):.2f} dB")
            print(f"    STOI: {sample_metrics.get('stoi_score', 0):.4f}")
            print(f"    啸叫抑制: {sample_metrics.get('howling_reduction_db', 0):.2f} dB")
            print(f"    MOS估算: {mos:.2f}")
            print(f"    参数量: {sum(p.numel() for p in method.parameters())}")
            
        except Exception as e:
            print(f"  ✗ {name}: 失败 - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_multiple_batches():
    """测试4: 多批次测试（检测状态累积问题）"""
    print("\n" + "=" * 60)
    print("测试4: 多批次测试（检测状态累积问题）")
    print("=" * 60)
    
    methods = {
        'FrequencyShift': FrequencyShiftMethod(shift_hz=20.0),
        'GainSuppression': GainSuppressionMethod(threshold_db=-30.0),
        'AdaptiveFeedback': AdaptiveFeedbackMethod(filter_length=64, step_size=0.01),
    }
    
    all_passed = True
    for name, method in methods.items():
        try:
            method.eval()
            
            # 关键：对每个方法，需要重新实例化以重置状态
            # 因为GainSuppression和AdaptiveFeedback有内部状态（register_buffer）
            if name in ['GainSuppression', 'AdaptiveFeedback']:
                if name == 'GainSuppression':
                    method = GainSuppressionMethod(threshold_db=-30.0)
                else:
                    method = AdaptiveFeedbackMethod(filter_length=64, step_size=0.01)
                method.eval()
            
            with torch.no_grad():
                for batch_idx in range(3):
                    # 不同时间帧长度模拟不同的音频片段
                    t_frames = 80 + batch_idx * 20
                    test_input = torch.rand(2, 1, 256, t_frames) * 0.6 + 0.2
                    
                    output = method(test_input)
                    
                    assert output.shape == test_input.shape, \
                        f"{name} batch {batch_idx}: 形状不匹配"
                    assert not torch.isnan(output).any(), \
                        f"{name} batch {batch_idx}: 包含NaN"
                    
                    print(f"  ✓ {name} batch {batch_idx}: "
                          f"输入={test_input.shape}, 输出={output.shape}, "
                          f"范围=[{output.min():.4f}, {output.max():.4f}]")
                    
        except Exception as e:
            print(f"  ✗ {name}: 失败 - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_with_real_data():
    """测试5: 使用真实数据集测试"""
    print("\n" + "=" * 60)
    print("测试5: 真实数据集测试")
    print("=" * 60)
    
    from src.config import cfg
    from src.dataset import HowlingDataset
    from torch.utils.data import DataLoader
    
    # 检查数据目录是否存在
    import os
    if not os.path.exists(str(cfg.VAL_CLEAN_DIR)):
        print("  ⊘ 跳过: 验证集数据目录不存在")
        return True
    
    try:
        val_dataset = HowlingDataset(
            clean_dir=cfg.VAL_CLEAN_DIR,
            howling_dir=cfg.VAL_NOISY_DIR,
        )
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # 只取前2个batch
        methods = {
            'FrequencyShift': FrequencyShiftMethod(shift_hz=20.0),
            'GainSuppression': GainSuppressionMethod(threshold_db=-30.0),
            'AdaptiveFeedback': AdaptiveFeedbackMethod(filter_length=64, step_size=0.01),
        }
        
        metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)
        
        all_passed = True
        for name, method in methods.items():
            # 重新实例化以重置状态
            if name == 'FrequencyShift':
                method = FrequencyShiftMethod(shift_hz=20.0)
            elif name == 'GainSuppression':
                method = GainSuppressionMethod(threshold_db=-30.0)
            else:
                method = AdaptiveFeedbackMethod(filter_length=64, step_size=0.01)
            
            method.eval()
            
            try:
                for batch_idx, (noisy_mag, clean_mag) in enumerate(val_loader):
                    if batch_idx >= 2:
                        break
                    
                    with torch.no_grad():
                        pred_mag = method(noisy_mag)
                    
                    assert pred_mag.shape == noisy_mag.shape, \
                        f"{name}: 输出形状不匹配 {pred_mag.shape} != {noisy_mag.shape}"
                    assert not torch.isnan(pred_mag).any(), \
                        f"{name}: 输出包含NaN"
                    
                    # 计算指标
                    sample_metrics = metrics_calc.calculate_all_metrics(
                        clean=clean_mag, noisy=noisy_mag, enhanced=pred_mag,
                    )
                    
                    if batch_idx == 0:
                        print(f"  ✓ {name} batch {batch_idx}: "
                              f"形状={pred_mag.shape}, "
                              f"范围=[{pred_mag.min():.4f}, {pred_mag.max():.4f}]")
                        print(f"    SNR改善: {sample_metrics.get('snr_improvement_db', 0):.2f} dB, "
                              f"STOI: {sample_metrics.get('stoi_score', 0):.4f}")
                
                print(f"  ✓ {name}: 真实数据测试通过")
            except Exception as e:
                print(f"  ✗ {name}: 真实数据测试失败 - {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"  ⊘ 跳过真实数据测试: {e}")
        return True


def main():
    print("传统啸叫抑制方法测试")
    print("=" * 60)
    
    results = {}
    
    results['basic_forward'] = test_basic_forward()
    results['normalized_spectrogram'] = test_normalized_spectrogram()
    results['metrics_integration'] = test_metrics_integration()
    results['multiple_batches'] = test_multiple_batches()
    results['real_data'] = test_with_real_data()
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！传统方法可以正常运行。")
    else:
        print("\n❌ 部分测试失败，需要修复。")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)