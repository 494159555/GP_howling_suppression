"""生成最终实验结果报告

合并所有评估结果到统一文件中。
"""
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

# ===== 从各次运行中汇总结果 =====
all_results = {
    "Unprocessed": {
        "model_name": "Unprocessed (基线)",
        "avg_loss": 0.0230,
        "snr_improvement_db": 0.0,
        "psnr_db": 23.49,
        "stoi_score": 0.9601,
        "howling_reduction_db": 0.0,
        "mos_estimate": 3.40,
        "avg_inference_ms": 0.0,
        "param_count": 0,
        "eval_samples": 655,
        "note": "未处理基线，直接使用带啸叫信号"
    },
    "unet_v1": {
        "model_name": "AudioUNet3",
        "avg_loss": 0.0152,
        "snr_improvement_db": 45.45,
        "psnr_db": 32.32,
        "stoi_score": 1.0000,
        "howling_reduction_db": -80.0,
        "mos_estimate": 4.60,
        "avg_inference_ms": 20.88,
        "param_count": 51601,
        "eval_samples": 655,
        "note": "3层U-Net (轻量级基线), 100 epochs"
    },
    "unet_v2": {
        "model_name": "AudioUNet5",
        "avg_loss": 0.0138,
        "snr_improvement_db": 45.96,
        "psnr_db": 32.61,
        "stoi_score": 1.0000,
        "howling_reduction_db": -80.0,
        "mos_estimate": 4.60,
        "avg_inference_ms": 30.50,
        "param_count": 882769,
        "eval_samples": 655,
        "note": "5层U-Net (默认模型), 100 epochs"
    },
    "unet_v3_attention": {
        "model_name": "AudioUNet5Attention",
        "avg_loss": 0.0138,
        "snr_improvement_db": 46.03,
        "psnr_db": 32.61,
        "stoi_score": 1.0000,
        "howling_reduction_db": -80.0,
        "mos_estimate": 4.60,
        "avg_inference_ms": 39.08,
        "param_count": 905381,
        "eval_samples": 655,
        "note": "5层U-Net + 注意力门, 100 epochs"
    },
    "unet_v6_optimized": {
        "model_name": "AudioUNet5Optimized",
        "avg_loss": 0.0132,
        "snr_improvement_db": 45.84,
        "psnr_db": 32.83,
        "stoi_score": 1.0000,
        "howling_reduction_db": -80.0,
        "mos_estimate": 4.60,
        "avg_inference_ms": 73.55,
        "param_count": 3132515,
        "eval_samples": 655,
        "note": "5层U-Net + 注意力+残差+空洞(综合优化), 100 epochs"
    },
    "unet_v10_gan": {
        "model_name": "AudioUNet5GAN",
        "avg_loss": 0.1888,
        "snr_improvement_db": 66.83,
        "psnr_db": 22.40,
        "stoi_score": 1.0000,
        "howling_reduction_db": -80.0,
        "mos_estimate": 4.60,
        "avg_inference_ms": 30.50,
        "param_count": 1540754,
        "eval_samples": 655,
        "note": "5层U-Net + GAN框架, 100 epochs (仅生成器评估)"
    },
    "FrequencyShift": {
        "model_name": "FrequencyShift",
        "description": "移频法 — 频偏 20Hz，STFT域线性插值",
        "avg_loss": 0.0578,
        "snr_improvement_db": 41.57,
        "psnr_db": 26.51,
        "stoi_score": 1.0000,
        "howling_reduction_db": -80.0,
        "mos_estimate": 4.60,
        "avg_inference_ms": 21.42,
        "param_count": 0,
        "eval_samples": 33,
        "note": "无参数方法, 真实数据子集评估"
    },
    "GainSuppression": {
        "model_name": "GainSuppression",
        "description": "增益抑制法 — 频率范围 1kHz-8kHz，阈值 -30dB，衰减 -20dB",
        "avg_loss": 0.3146,
        "snr_improvement_db": 53.70,
        "psnr_db": 17.85,
        "stoi_score": 0.9513,
        "howling_reduction_db": -80.0,
        "mos_estimate": 4.60,
        "avg_inference_ms": 93353.46,
        "param_count": 0,
        "eval_samples": 3,
        "note": "无参数方法, 合成数据评估(实现含逐帧Python循环, 大数据集耗时过长)"
    },
    "AdaptiveFeedback": {
        "model_name": "AdaptiveFeedback",
        "description": "自适应反馈消除法 — NLMS算法，滤波器长度64，步长0.01",
        "avg_loss": 0.1890,
        "snr_improvement_db": 53.33,
        "psnr_db": 20.37,
        "stoi_score": 0.9728,
        "howling_reduction_db": -80.0,
        "mos_estimate": 4.60,
        "avg_inference_ms": 45.53,
        "param_count": 0,
        "eval_samples": 3,
        "note": "无参数方法, 合成数据评估"
    },
}

# ===== 保存统一结果 =====
output_dir = PROJECT_ROOT / 'experiment_results'

# 主结果文件
with open(output_dir / 'evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)

# 实验2.2: 传统方法结果
trad_results = {k: all_results[k] for k in ['FrequencyShift', 'GainSuppression', 'AdaptiveFeedback']}
trad_dir = output_dir / 'exp2_2_traditional'
trad_dir.mkdir(parents=True, exist_ok=True)
with open(trad_dir / 'traditional_results.json', 'w', encoding='utf-8') as f:
    json.dump(trad_results, f, indent=4, ensure_ascii=False, default=str)

# 实验3: 统一评估结果
unified_dir = output_dir / 'exp3_unified_eval'
unified_dir.mkdir(parents=True, exist_ok=True)
with open(unified_dir / 'unified_results.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)

# ===== 打印表格 =====
def print_table(title, methods, results):
    header = (f"{'方法':25s} | {'参数量':>10s} | {'Loss':>8s} | "
              f"{'SNR(dB)':>8s} | {'STOI':>7s} | {'啸叫抑制(dB)':>10s} | "
              f"{'MOS':>5s} | {'推理(ms)':>10s}")
    sep = "-" * len(header)
    print(f"\n{'='*105}")
    print(f"  {title}")
    print(f"{'='*105}")
    print(header)
    print(sep)
    for m in methods:
        if m in results:
            r = results[m]
            name = r.get('model_name', m)
            print(f"{name:25s} | {r['param_count']:>10,d} | "
                  f"{r['avg_loss']:8.4f} | {r['snr_improvement_db']:8.2f} | "
                  f"{r['stoi_score']:7.4f} | {r['howling_reduction_db']:10.2f} | "
                  f"{r['mos_estimate']:5.2f} | {r['avg_inference_ms']:10.2f}")

print_table("表5-2 传统方法评估结果",
            ['FrequencyShift', 'GainSuppression', 'AdaptiveFeedback'], all_results)

dl_order = ['Unprocessed', 'unet_v1', 'unet_v2', 'unet_v3_attention', 'unet_v6_optimized', 'unet_v10_gan']
print_table("表5-3 深度学习模型评估结果", dl_order, all_results)

print_table("综合排名（MOS降序）",
            [k for k, _ in sorted(all_results.items(), key=lambda x: x[1]['mos_estimate'], reverse=True)],
            all_results)

print(f"\n结果已保存:")
print(f"  {output_dir / 'evaluation_results.json'}")
print(f"  {trad_dir / 'traditional_results.json'}")
print(f"  {unified_dir / 'unified_results.json'}")
