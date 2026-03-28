# 配置文件说明

本目录包含所有U-Net模型变体的训练配置文件（YAML格式）。

---

## 📁 配置文件完整列表

### 基线模型（v1-v2）

| 配置文件 | 模型名称 | 描述 | 适用场景 |
|---------|---------|------|---------|
| `unet_v1.yaml` | AudioUNet3 | 3层基线U-Net | 快速原型开发 |
| `unet_v2.yaml` | AudioUNet5 | 5层基线U-Net | 通用场景（默认） |

### 改进模型（v3-v6）

| 配置文件 | 模型名称 | 描述 | 特点 |
|---------|---------|------|------|
| `unet_v3_attention.yaml` | AudioUNet5Attention | 注意力机制U-Net | 关注重要频段 |
| `unet_v4_residual.yaml` | AudioUNet5Residual | 残差连接U-Net | 更深的网络 |
| `unet_v5_dilated.yaml` | AudioUNet5Dilated | 空洞卷积U-Net | 大感受野 |
| `unet_v6_optimized.yaml` | AudioUNet5Optimized | 综合优化U-Net | 注意力+残差+空洞（推荐） |

### 时序模型（v7-v9）

| 配置文件 | 模型名称 | 描述 | 适用场景 |
|---------|---------|------|---------|
| `unet_v7_lstm.yaml` | AudioUNet5LSTM | 双向LSTM U-Net | 时变啸叫 |
| `unet_v8_temporal_attention.yaml` | AudioUNet5TemporalAttention | 时间注意力U-Net | 关注特定时段 |
| `unet_v9_convlstm.yaml` | AudioUNet5ConvLSTM | ConvLSTM U-Net | 时空建模 |

### 高级模型（v10-v13）

| 配置文件 | 模型名称 | 描述 | 特点 |
|---------|---------|------|------|
| `unet_v10_gan.yaml` | AudioUNet5GAN | GAN架构U-Net | 最佳生成质量 |
| `unet_v11_multiscale.yaml` | AudioUNet5MultiScale | 多尺度U-Net | 多尺度特征 |
| `unet_v12_pyramid.yaml` | AudioUNet5Pyramid | 金字塔池化U-Net | 全局上下文 |
| `unet_v13_fpn.yaml` | AudioUNet5FPN | 特征金字塔U-Net | 强语义+精细节 |

### 训练策略配置（v14-v16）

| 配置文件 | 描述 | 适用场景 |
|---------|------|---------|
| `unet_v14_mixed_precision.yaml` | 混合精度训练（AMP） | GPU加速训练 |
| `unet_v15_curriculum.yaml` | 课程学习策略 | 稳定训练收敛 |
| `unet_v16_lr_scheduler.yaml` | 高级学习率调度 | 优化收敛效果 |

---

## 🚀 使用方法

### 方式1：使用配置文件（推荐）

```bash
# 基线模型
python src/train.py --config configs/unet_v2.yaml

# 注意力机制模型
python src/train.py --config configs/unet_v3_attention.yaml

# 综合优化模型（推荐）
python src/train.py --config configs/unet_v6_optimized.yaml

# 时序模型
python src/train.py --config configs/unet_v7_lstm.yaml

# 高级模型
python src/train.py --config configs/unet_v13_fpn.yaml
```

### 方式2：命令行参数

```bash
# 直接指定模型
python src/train.py --model unet_v6_optimized

# 指定模型并修改参数
python src/train.py --model unet_v6_optimized --lr 2e-4 --batch-size 4 --epochs 100

# 自定义实验名称
python src/train.py --model unet_v6_optimized --exp-name "my_experiment"
```

### 方式3：配置文件 + 命令行覆盖

```bash
# 使用配置文件，但覆盖部分参数
python src/train.py --config configs/unet_v3_attention.yaml --epochs 100 --lr 5e-5
```

---

## 📝 配置文件格式说明

### 标准配置结构

```yaml
# ========== 模型配置 ==========
model:
  name: unet_v2              # 模型标识符
  class_name: AudioUNet5     # 类名
  description: "5层基线U-Net"  # 描述

# ========== 训练参数 ==========
training:
  learning_rate: 0.0001      # 学习率
  batch_size: 8              # 批大小
  num_epochs: 50             # 训练轮数
  num_workers: 2             # 数据加载线程数
  weight_decay: 1.0e-5       # 权重衰减

  # 学习率调度器
  lr_scheduler:
    type: "plateau"          # 类型: plateau, cosine_warmup, one_cycle, step
    factor: 0.5
    patience: 3

# ========== 损失函数配置 ==========
loss:
  type: "l1"                 # 类型: l1, mse, spectral, multitask, adversarial

# ========== 数据增强配置 ==========
augmentation:
  enabled: false             # 是否启用
  spec_augment:
    freq_masking: true
    time_masking: true

# ========== 后处理配置 ==========
post_processing:
  enabled: false

# ========== 实验信息 ==========
experiment:
  name: "baseline_unet_v2"
  description: "5层U-Net基线模型"
```

---

## 💡 配置优先级

```
命令行参数 > YAML配置文件 > src/config.py默认值
```

例如：
```bash
python src/train.py --config configs/unet_v2.yaml --lr 0.001 --batch-size 16

# 实际使用：lr=0.001, batch_size=16 (命令行优先)
# 其他参数：从unet_v2.yaml读取
# 未指定参数：从src/config.py读取默认值
```

---

## 🎯 模型选择建议

### 快速原型开发
```bash
python src/train.py --config configs/unet_v1.yaml
```
- 参数量最少，训练最快
- 适合快速验证想法

### 标准使用（推荐）
```bash
python src/train.py --config configs/unet_v2.yaml
```
- 默认模型，性能均衡
- 适合大多数场景

### 追求性能
```bash
python src/train.py --config configs/unet_v6_optimized.yaml
```
- 综合优化版本
- 注意力+残差+空洞卷积

### 时变啸叫问题
```bash
python src/train.py --config configs/unet_v7_lstm.yaml
# 或
python src/train.py --config configs/unet_v9_convlstm.yaml
```
- 时序建模能力强
- 适合处理时变啸叫

### GPU训练加速
```bash
python src/train.py --config configs/unet_v14_mixed_precision.yaml
```
- 混合精度训练
- 速度提升约1.5-2倍

---

## 🔧 创建自定义配置

### 步骤1：复制现有配置

```bash
cp configs/unet_v2.yaml configs/my_custom_config.yaml
```

### 步骤2：修改参数

```yaml
# 编辑 my_custom_config.yaml
model:
  name: unet_v2
  description: "我的自定义配置"

training:
  learning_rate: 0.0002      # 修改学习率
  batch_size: 16             # 增大批大小
  num_epochs: 100            # 增加训练轮数

experiment:
  name: "my_experiment"      # 自定义实验名称
```

### 步骤3：使用自定义配置

```bash
python src/train.py --config configs/my_custom_config.yaml
```

---

## 📊 配置参数说明

### 学习率调度器类型

| 类型 | 说明 | 适用场景 |
|-----|------|---------|
| `plateau` | 验证Loss不下降时降低学习率 | 通用场景 |
| `cosine_warmup` | 余弦退火+预热策略 | 追求最佳收敛 |
| `one_cycle` | 单周期策略 | 快速训练 |
| `step` | 固定间隔衰减 | 简单场景 |

### 损失函数类型

| 类型 | 说明 |
|-----|------|
| `l1` | L1距离损失，简单有效 |
| `mse` | 均方误差损失 |
| `spectral` | 频谱损失，关注频域 |
| `multitask` | 多任务组合损失 |
| `adversarial` | 对抗损失（GAN） |

---

## ❓ 常见问题

### Q: 找不到配置文件？
**A**: 确保在项目根目录运行，或使用绝对路径

### Q: 配置不生效？
**A**: 检查参数优先级，命令行参数会覆盖配置文件

### Q: 显存不足？
**A**: 减小`batch_size`，或使用混合精度配置（v14）

### Q: 训练不收敛？
**A**: 尝试：
1. 降低学习率
2. 使用课程学习配置（v15）
3. 使用高级学习率调度（v16）

---

## 🔍 查看帮助

```bash
# 查看训练脚本帮助
python src/train.py --help

# 查看配置文件
cat configs/unet_v2.yaml

# 对比不同配置
diff configs/unet_v2.yaml configs/unet_v6_optimized.yaml
```

---

## 📝 维护说明

- 所有配置文件使用统一的格式
- 使用中文注释，方便理解
- 每个配置都有详细的字段说明
- 参数命名使用下划线风格（如 `num_epochs`）

**最后更新**: 2025-03-26
