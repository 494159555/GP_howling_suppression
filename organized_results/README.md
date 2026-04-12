# 声反馈啸叫抑制实验 — 整理结果总览

> 基于《基于深度U-Net的声反馈啸叫抑制方法》论文（第3-5章）
> 整理日期：2026-04-06

## 目录结构

```
organized_results/
├── README.md                          ← 本文件（总索引）
├── exp1_model_training/               ← 实验1：5个U-Net模型训练
│   ├── summary.md                     ← 结果摘要
│   └── training_summary.json          ← 原始训练数据
├── exp2_traditional_methods/          ← 实验2.2：传统方法评估
│   ├── summary.md                     ← 结果摘要
│   └── traditional_results.json       ← 原始评估数据
├── exp3_unified_evaluation/           ← 实验3：统一评估（全部方法）
│   ├── summary.md                     ← 结果摘要（核心对比）
│   ├── evaluation_results.json        ← 完整评估数据
│   └── unified_results.json           ← 统一评估结果
├── exp4_loss_comparison/              ← 实验4：损失函数对比
│   ├── summary.md                     ← 结果摘要
│   ├── loss_comparison_results.json   ← 原始对比数据
│   └── table5_5_loss_comparison.md    ← 论文表格（表5-5）
├── exp5_training_strategy/            ← 实验5：训练策略对比
│   ├── summary.md                     ← 结果摘要
│   ├── strategy_comparison_results.json ← 原始对比数据
│   └── experiment5_report.md          ← 详细分析报告
└── exp6_augmentation_comparison/      ← 实验6-7：数据增强对比
    ├── summary.md                     ← 结果摘要
    ├── exp6_augmentation_results.json ← 实验6原始数据
    ├── exp7_augmentation_results.json ← 实验7原始数据
    ├── exp6_report.md                 ← 实验6分析报告
    └── exp7_report.md                 ← 实验7分析报告
```

## 实验完成状态

| 实验 | 状态 | 说明 |
|:---|:---:|:---|
| 实验1: 模型训练 (5个U-Net) | 已完成 | V1/V2/V3/V6/V10 全部训练成功 |
| 实验2.2: 传统方法 | 已完成 | 移频法/增益抑制/自适应反馈消除 |
| 实验3: 统一评估 | 已完成 | 655个测试样本，9种方法对比 |
| 实验4: 损失函数对比 | 已完成 | MSE/L1/SI-SDR/MR-STFT/Composite |
| 实验5: 训练策略对比 | 已完成 | 5种LR调度策略 |
| 实验6-7: 数据增强对比 | 已完成 | 5种增强策略，结论：帮助有限 |
| 实验3(消融实验) | 未完成 | EXPERIMENTS.md中的消融实验无独立结果 |

## 核心结论速览

### 最佳模型: AudioUNet5GAN (V10)
- SNR提升: **66.83 dB** (最高)
- 参数量: 1.54M
- 推理速度: 30.50 ms

### 最佳损失函数: Composite Loss (MR-STFT + SI-SDR)
- SNR提升: **71.00 dB**

### 最佳训练策略: WarmupCosine
- 训练效率最高: 2994s / 71 epochs
- SNR提升: 70.93 dB

### 最佳增强策略: 无增强 (Baseline)
- 增强对当前任务帮助有限

### 传统方法 vs 深度学习
- 传统方法最佳 SNR: 53.70 dB (增益抑制法)
- 深度学习最佳 SNR: 66.83 dB (AudioUNet5GAN)
- 深度学习方法在所有指标上全面优于传统方法
