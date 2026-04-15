# 表5-7: 数据增强对比结果

**模型**: AudioUNet5Optimized (V6)  
**损失函数**: Composite Loss (spectral=0.5, l1=0.3, mse=0.2)  
**学习率调度**: Warmup + CosineDecay (warmup=5 epochs, lr=1e-3 → 1e-6)  

| 增强策略 | Best Val Loss | L1 Loss | SNR改善(dB) | STOI | 啸叫抑制(dB) | MOS | 训练耗时(s) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| FullAugmentation | 0.0099 | 0.0128 | 23.68 | 0.9573 | -11.62 | 3.87 | 2022.1 |
