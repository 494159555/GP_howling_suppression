# 声反馈啸叫抑制项目 — 实验清单

> 基于《基于深度U-Net的声反馈啸叫抑制方法》论文（第3-5章）整理的全部实验任务。
> 代码文件路径基于项目根目录，状态标注为 `[ ]` 待完成 / `[x]` 已完成。

---

## 阶段一：数据准备

### 1.1 基于 RIR 的啸叫仿真数据生成 （已完成）
- [ ] 配置 RIR 生成器参数（图像法）
  - 房间尺寸：随机生成
  - 混响时间 RT60：0.1s ~ 0.5s 随机
  - 麦克风/扬声器位置：随机
- [ ] 配置系统参数
  - 系统延迟 Δt：0.02s ~ 0.2s 随机
  - 增益 G：5dB ~ 10dB 随机
- [ ] 实现非线性失真模拟（硬限幅 + Sigmoid）
- [ ] 实现逐帧声反馈仿真流程（增益 → 非线性 → RIR卷积 → 叠加）
- [ ] 生成干净语音-带啸叫语音配对数据集
- [ ] 划分训练集、验证集、测试集（测试集 RIR 参数独立于训练集）
- [ ] 验证数据质量（抽样检查频谱、试听）

**关键文件**: `src/dataset.py`, 数据路径: `/mnt/ent_disk0/syx/howling_data/train|dev`
**STFT 参数**: FFT=512, Hop=128/256, Hamming窗, 采样率 16kHz

---

## 阶段二：模型训练与对比（实验1）

### 2.1 训练深度学习模型（5个U-Net变体）（已完成）

| 编号 | 模型 | 参数量 | 关键特性 | 配置文件 |
|:---:|:---|:---:|:---|:---|
| V1 | AudioUNet3 | ~1.2M | 3层编码器-解码器，基线 | `configs/unet_v1.yaml` |
| V2 | AudioUNet5 | ~4.5M | 5层编码器-解码器，默认模型 | `configs/unet_v2.yaml` |
| V3 | AudioUNet5Attention | ~4.8M | V2 + 跳跃连接注意力门 | `configs/unet_v3_attention.yaml` |
| V6 | AudioUNet5Optimized | ~6.2M | V3 + 残差连接 + 空洞卷积(ASPP) | `configs/unet_v6_optimized.yaml` |
| V10 | AudioUNet5GAN | ~5.1M(生成器) | V2 + GAN对抗训练(LSGAN) | `configs/unet_v10_gan.yaml` |

- [ ] V1: AudioUNet3 训练
- [ ] V2: AudioUNet5 训练
- [ ] V3: AudioUNet5Attention 训练
- [ ] V6: AudioUNet5Optimized 训练
- [ ] V10: AudioUNet5GAN 训练（含生成器+判别器交替训练）

**统一训练参数**: batch_size=16, lr=1e-3, epochs=100, early_stopping=15, Adam优化器, AMP混合精度, 梯度裁剪=1.0, weight_decay=1e-5
**训练脚本**: `src/train.py` 或 `src/train_v2.py`

### 2.2 运行传统基线方法

- [ ] 移频法（FrequencyShiftMethod）— 频偏 20Hz，STFT域线性插值
- [ ] 增益抑制法（GainSuppressionMethod）— 频率范围 1kHz-8kHz，阈值 -30dB，衰减 -20dB
- [ ] 自适应反馈消除法（AdaptiveFeedbackMethod）— NLMS算法，滤波器长度64，步长0.01

**实现文件**: `src/traditional/frequency_shift.py`, `gain_suppression.py`, `adaptive_feedback.py`

---

## 阶段三：统一评估（实验2）

- [ ] 在测试集上对所有方法统一推理
  - 5个深度学习模型 + 3个传统方法 + 未处理基线
- [ ] 计算客观评价指标
  - SI-SDR（尺度不变信噪比）
  - PESQ（语音质量感知评估，0.5~4.5）
  - STOI（短时客观可懂度，0~1）
- [ ] 记录推理时间（ms/条）
- [ ] 统计模型参数量
- [ ] 生成对比表格（表5-2 传统方法、表5-3 深度学习模型）

**评估脚本**: `scripts/evaluate_all.py`, `src/evaluate.py`
**指标实现**: `src/evaluation/metrics.py`

---

## 阶段四：消融实验（实验3）

基于 AudioUNet5Optimized (V6)，逐个移除组件验证独立贡献：

| 配置 | 注意力门 | 残差连接 | 空洞卷积 |
|:---:|:---:|:---:|:---:|
| Full（完整） | ✓ | ✓ | ✓ |
| w/o Attention | ✗ | ✓ | ✓ |
| w/o Residual | ✓ | ✗ | ✓ |
| w/o Dilated | ✓ | ✓ | ✗ |
| Baseline（全移除） | ✗ | ✗ | ✗ |

- [ ] Full 完整模型训练与评估（已有V6结果可复用）
- [ ] w/o Attention 移除注意力门，重新训练
- [ ] w/o Residual 移除残差连接，重新训练
- [ ] w/o Dilated 移除空洞卷积（膨胀率统一为1），重新训练
- [ ] Baseline 全部移除，重新训练（等效于 AudioUNet5）
- [ ] 汇总5组消融结果，生成消融对比表（表5-4）
- [ ] 分析组件协同效应

**消融脚本**: `scripts/ablation_study.py`

---

## 阶段五：损失函数对比（实验4）

固定模型为 AudioUNet5Optimized (V6)，仅替换损失函数：

| 序号 | 损失函数 | 说明 |
|:---:|:---|:---|
| 1 | MSE Loss | 均方误差，基础回归损失 |
| 2 | L1 Loss | 平均绝对误差，对异常值不敏感 |
| 3 | SI-SDR Loss | 尺度不变信噪比负值，信号级损失 |
| 4 | Multi-Resolution STFT Loss | 多分辨率(帧长512/256/128)频谱L1损失 |
| 5 | Composite Loss | 多分辨率STFT(α=1.0) + SI-SDR(β=0.5) |

- [ ] MSE Loss 训练 + 评估
- [ ] L1 Loss 训练 + 评估
- [ ] SI-SDR Loss 训练 + 评估
- [ ] Multi-Resolution STFT Loss 训练 + 评估
- [ ] Composite Loss 训练 + 评估
- [ ] 汇总5组结果，生成损失函数对比表（表5-5）

**对比脚本**: `scripts/loss_comparison.py`

---

## 阶段六：训练策略对比（实验5）（已完成）

固定模型为 AudioUNet5Optimized (V6)，损失函数为 Composite Loss，对比学习率调度策略：

| 序号 | 策略 | 关键参数 |
|:---:|:---|:---|
| 1 | CosineAnnealingLR | lr: 1e-3 → 1e-6，余弦衰减 |
| 2 | ReduceLROnPlateau | patience=5, factor=0.5, min_lr=1e-6 |
| 3 | CyclicLR | lr: 1e-5 ~ 1e-3，三角循环 |
| 4 | OneCycleLR | 前30%升至1e-3，后70%退火至1e-6 |
| 5 | Warmup + CosineDecay | 前5轮线性预热至1e-3，后余弦衰减至1e-6 |

- [x] CosineAnnealingLR 训练 + 评估
- [x] ReduceLROnPlateau 训练 + 评估
- [x] CyclicLR 训练 + 评估
- [x] OneCycleLR 训练 + 评估
- [x] Warmup + CosineDecay 训练 + 评估
- [x] 汇总5组结果，生成训练策略对比表（表5-6）

**对比脚本**: `scripts/exp5_training_strategy.py`
**结果目录**: `experiments/exp5_training_strategy/`

---

## 阶段七：数据增强对比（实验6）

固定模型为 AudioUNet5Optimized (V6)，Composite Loss + Warmup+CosineDecay，对比增强策略：

| 序号 | 增强策略 | 说明 |
|:---:|:---|:---|
| 1 | 无增强（Baseline） | 原始STFT特征 |
| 2 | 频率掩蔽（FreqMask） | 最大宽度20频率bin，2个掩码 |
| 3 | 时间掩蔽（TimeMask） | 最大宽度20时间帧，2个掩码 |
| 4 | 联合掩蔽（JointMask） | 频率 + 时间掩蔽同时应用 |
| 5 | 综合增强（Full Augmentation） | 联合掩蔽 + 增益缩放(0.8~1.2) + 噪声注入(SNR 20~40dB) + Mixup(α=0.4) |

- [ ] 无增强训练 + 评估
- [ ] 频率掩蔽训练 + 评估
- [ ] 时间掩蔽训练 + 评估
- [ ] 联合掩蔽训练 + 评估
- [ ] 综合增强训练 + 评估
- [ ] 汇总5组结果，生成数据增强对比表（表5-7）

**对比脚本**: `scripts/augmentation_comparison.py`
**增强实现**: `src/models/augmentation.py`

---

## 阶段八：结果可视化

### 8.1 频谱对比图
- [ ] 典型样本的频谱对比：干净语音 / 带啸叫 / 各方法处理后（图5-1）
- [ ] 覆盖方法：移频法、自适应反馈消除法、AudioUNet3、AudioUNet5、AudioUNet5Optimized、AudioUNet5GAN

### 8.2 训练曲线
- [ ] 5个模型的训练损失曲线（图5-3上）
- [ ] 5个模型的验证 SI-SDR 曲线（图5-3下）

### 8.3 模型性能对比图
- [ ] 模型性能演进折线图（图5-2）：SI-SDR / PESQ / STOI

### 8.4 雷达图
- [ ] 各模型多维性能雷达图（SI-SDR、PESQ、STOI、推理时间、参数量）

### 8.5 消融实验热力图
- [ ] 各组件在不同指标上的贡献度热力图

### 8.6 损失函数对比柱状图
- [ ] 分组柱状图 + 误差棒（图5-5对应）

### 8.7 学习率调度对比曲线
- [ ] 各策略的学习率变化曲线
- [ ] 对应的验证损失变化曲线

**可视化脚本**: `src/evaluation/visualizer.py`

---

## 实验总量统计

| 类别 | 数量 |
|:---|:---:|
| 数据生成任务 | 7 |
| 模型训练（5个模型） | 5 |
| 传统方法运行 | 3 |
| 统一评估 | 1 |
| 消融实验训练+评估 | 4（Full可复用） |
| 损失函数对比训练+评估 | 5 |
| 训练策略对比训练+评估 | 5 |
| 数据增强对比训练+评估 | 5 |
| 可视化任务 | 7 |
| **总计** | **~42项** |

> 注：消融实验的 Full 配置可复用阶段二的 V6 训练结果，实际独立训练次数约 20 次左右。
