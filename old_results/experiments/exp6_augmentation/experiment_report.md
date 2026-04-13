# 实验6：数据增强策略对比 — 实验报告

## 实验配置

| 项目 | 配置 |
|:---|:---|
| 模型 | AudioUNet5Optimized (V6) |
| 参数量 | 3,132,515 |
| 损失函数 | CompositeLoss (Multi-Resolution STFT α=1.0 + SI-SDR β=0.5) |
| 学习率调度 | Warmup(5 epochs) + CosineDecay (1e-3 → 1e-6) |
| 优化器 | Adam (weight_decay=1e-5, grad_clip=1.0) |
| 训练轮数 | 100 |
| 批大小 | 16 |
| 训练集大小 | 5,238 |
| 验证集大小 | 655 |
| 采样率 | 16kHz |
| STFT参数 | FFT=512, Hop=128, Hamming窗 |

## 增强策略说明

| 序号 | 策略 | 说明 |
|:---:|:---|:---|
| 1 | 无增强(Baseline) | 原始STFT特征，不做任何增强 |
| 2 | 频率掩蔽(FreqMask) | 最大宽度20频率bin，2个掩码 |
| 3 | 时间掩蔽(TimeMask) | 最大宽度20时间帧，2个掩码 |
| 4 | 联合掩蔽(JointMask) | 频率掩蔽 + 时间掩蔽同时应用 |
| 5 | 综合增强(Full) | 联合掩蔽 + 增益缩放(0.8~1.2) + 噪声注入(SNR 20~40dB) + Mixup(α=0.4) |

## 实验结果

### 表5-7 数据增强策略对比

| 增强策略 | L1 Loss ↓ | SNR改善(dB) ↑ | STOI ↑ | MOS ↑ | Howling衰减(dB) | 训练耗时(s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 无增强(Baseline) | 0.3039 | 70.92 | 1.0000 | 4.60 | -0.44 | 3976.7 |
| 综合增强(Full) | 0.1742 | 66.35 | 1.0000 | 4.60 | -1.87 | 4036.1 |
| 时间掩蔽(TimeMask) | 0.1431 | 65.02 | 1.0000 | 4.60 | -2.29 | 4013.9 |
| 频率掩蔽(FreqMask) | 0.0960 | 62.89 | 1.0000 | 4.60 | -3.45 | 3987.2 |
| 联合掩蔽(JointMask) | 0.1082 | 62.03 | 1.0000 | 4.60 | -3.41 | 4017.9 |

### 结果分析

1. **SNR改善排名**: 无增强(Baseline) > 综合增强(Full) > 时间掩蔽(TimeMask) > 频率掩蔽(FreqMask) > 联合掩蔽(JointMask)

2. **L1 Loss排名**: 频率掩蔽(FreqMask) 最低(0.0960) < 联合掩蔽(JointMask)(0.1082) < 时间掩蔽(TimeMask)(0.1431) < 综合增强(Full)(0.1742) < 无增强(Baseline)(0.3039)

3. **STOI**: 所有策略均达到 1.0000

4. **关键发现**:
   - Baseline虽然SNR最高(70.92dB)，但L1 Loss也最高(0.3039)，说明模型在频谱重建精度上不如使用增强的策略
   - 频率掩蔽(FreqMask)在L1 Loss上表现最优(0.0960)，说明对频率轴的随机遮蔽有助于模型学习更鲁棒的特征
   - 联合掩蔽(JointMask)的效果略逊于单独的FreqMask或TimeMask，可能因为同时遮蔽频率和时间信息过多
   - 综合增强(Full)虽然SNR改善略低于Baseline，但L1 Loss明显更低，表明综合增强在信号质量上有优势

## 输出文件

```
experiments/exp6_augmentation/
├── augmentation_comparison_results.json  # 完整实验结果（JSON格式）
├── table_5_7_latex.txt                   # LaTeX格式表格
├── training_curves_data.json             # 训练曲线数据
├── training_curves.png                   # 训练/验证损失曲线图
├── metrics_comparison.png                # 指标对比柱状图
├── lr_schedule.png                       # 学习率调度曲线
├── radar_chart.png                       # 多维性能雷达图
└── experiment_report.md                  # 本报告
```
