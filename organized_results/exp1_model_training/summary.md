# 实验1：5个U-Net变体模型训练

> 训练时间：2026-03-30，总耗时 866.7 秒，5个模型全部训练成功

## 模型概览

| 编号 | 模型名 | 参数量 | 训练耗时(s) | 状态 |
|:---:|:---|:---:|:---:|:---:|
| V1 | AudioUNet3 (3层) | ~51.6K | 99.2 | 成功 |
| V2 | AudioUNet5 (5层) | ~882.8K | 138.2 | 成功 |
| V3 | AudioUNet5Attention | ~905.4K | 260.9 | 成功 |
| V6 | AudioUNet5Optimized | ~3.13M | 364.5 | 成功 |
| V10 | AudioUNet5GAN | ~1.54M | 3.9 | 成功 |

## 统一训练配置

- Epochs: 100 (各模型独立)
- Batch size: 64
- Learning rate: 1e-4
- Seed: 42

## 原始数据文件

- `training_summary.json` — 训练汇总数据
- 原始训练日志位于 `experiments/exp_20260330_*/`
