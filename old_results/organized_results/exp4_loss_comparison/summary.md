# 实验4：损失函数对比

> 固定模型 AudioUNet5Optimized (V6)，仅替换损失函数，训练50 epochs

## 损失函数对比结果（表5-5）

| 损失函数 | Best Val Loss | SNR提升(dB) | STOI | MOS | 训练时间(s) |
|:---|:---:|:---:|:---:|:---:|:---:|
| MSE Loss | 0.00334 | 47.00 | 1.0000 | 4.6 | 5640 |
| L1 Loss | 0.01282 | 46.14 | 1.0000 | 4.6 | 5803 |
| SI-SDR Loss | -31.008 | **71.14** | 1.0000 | 4.6 | 5783 |
| Multi-Resolution STFT | 0.01198 | 46.41 | 1.0000 | 4.6 | 5778 |
| Composite (MR-STFT+SI-SDR) | -15.215 | **71.00** | 1.0000 | 4.6 | 5783 |

## 关键发现

1. **SI-SDR Loss 和 Composite Loss** 表现远优于其他损失函数，SNR提升均超70dB
2. **MSE / L1 / MR-STFT** 三种损失函数性能接近，SNR提升约46-47dB
3. **Composite Loss** 综合了MR-STFT和SI-SDR的优势，是推荐选择
4. 所有损失函数的STOI和MOS指标相同，差异主要体现在SNR上

## 原始数据文件

- `loss_comparison_results.json` — 完整对比数据（含训练曲线）
- `table5_5_loss_comparison.md` — Markdown格式对比表
