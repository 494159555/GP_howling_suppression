# 实验5: 训练策略对比

## 实验设置

- **模型**: AudioUNet5Optimized (V6)
- **损失函数**: Composite Loss (Multi-Resolution STFT α=1.0 + SI-SDR β=0.5)
- **优化器**: Adam, weight_decay=1e-5
- **初始学习率**: 1e-3
- **批大小**: 16
- **最大轮数**: 100
- **早停耐心**: 15
- **混合精度**: 开启
- **梯度裁剪**: 1.0

## 对比策略

| 序号 | 策略 | 说明 |
|:---:|:---|:---|
| 1 | CosineAnnealingLR | lr: 1e-3 → 1e-6，余弦衰减 |
| 2 | ReduceLROnPlateau | patience=5, factor=0.5, min_lr=1e-6 |
| 3 | CyclicLR | lr: 1e-5 ~ 1e-3，三角循环 |
| 4 | OneCycleLR | 前30%升至1e-3，后70%退火至1e-6 |
| 5 | Warmup + CosineDecay | 前5轮线性预热至1e-3，后余弦衰减至1e-6 |

## 结果对比（表5-6）

| 策略 | Epochs | Best Val Loss | L1 Loss | SNR(dB) | STOI | MOS | 耗时(s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CosineAnnealing | 50 | 0.0697 | 0.0128 | 23.68 | 0.9583 | 3.88 | 1264.5 |

## 结论

**最佳策略**: CosineAnnealing (MOS=3.88)

- CosineAnnealing 在验证集上取得最佳MOS评分 3.88
- SNR提升: 23.68 dB
- STOI: 0.9583
- 训练耗时: 1264.5 秒 (50 epochs)

## 各策略训练曲线

### CosineAnnealing

- 最终训练Loss: 0.0697
- 最终验证Loss: 0.0697
- 初始LR: 1.00e-03
- 最终LR: 1.99e-06

