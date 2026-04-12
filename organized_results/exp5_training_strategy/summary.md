# 实验5：训练策略对比（学习率调度）

> 固定模型 AudioUNet5Optimized (V6)，Composite Loss，对比5种学习率调度策略

## 策略对比结果（表5-6）

| 策略 | Best Val Loss | SNR提升(dB) | 实际Epochs | 训练时间(s) | 效率(dB/s) |
|:---|:---:|:---:|:---:|:---:|:---:|
| CosineAnnealing | -15.0355 | 70.93 | 87 | 6942.7 | 0.0102 |
| ReduceLROnPlateau | -15.0138 | 70.91 | 53 | 4204.6 | **0.0169** |
| CyclicLR | **-15.0925** | 70.94 | 100 | 7938.8 | 0.0089 |
| OneCycleLR | -15.0682 | 70.94 | 93 | 7336.2 | 0.0097 |
| WarmupCosine | -15.0258 | 70.93 | 71 | 2994.0 | **0.0237** |

## 关键发现

1. **CyclicLR** 获得最佳验证损失 (-15.0925)，但训练时间最长 (7938s)
2. **WarmupCosine** 训练效率最高，71 epochs / 2994秒完成，SNR仅略低于最佳
3. **ReduceLROnPlateau** 收敛最快 (53 epochs)，但训练效率不如WarmupCosine
4. 所有策略SNR差异极小 (<0.03dB)，说明模型结构是性能的决定性因素
5. **推荐策略**: WarmupCosine（效率最高）或 CyclicLR（精度最优）

## 原始数据文件

- `strategy_comparison_results.json` — 完整对比数据（含学习率曲线）
- `experiment5_report.md` — 详细分析报告
