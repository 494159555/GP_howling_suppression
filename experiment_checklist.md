# 实验修复与重跑清单

> 基于 `evaluation_fix_plan.md` 中诊断的Bug，当前所有评估数据不可信
> 创建日期：2026-04-12

---

## 第一阶段：代码修复 ✅ 必须先完成

### 1.1 修复 `src/evaluate.py` 参数传递Bug（致命）

- [ ] **Bug位置**：`evaluate_with_metrics` 函数（约第148行）
  ```python
  # ❌ 当前（错误）
  sample_metrics = metrics_calc.calculate_all_metrics(
      clean=pred_mag,      # 错误
      noisy=noisy_mag,
      enhanced=pred_mag,   # 错误
  )
  # ✅ 修复为
  sample_metrics = metrics_calc.calculate_all_metrics(
      clean=clean_mag,     # 正确
      noisy=noisy_mag,
      enhanced=pred_mag,   # 正确
  )
  ```

- [ ] **Bug位置**：`evaluate_traditional_methods` 函数（约第232行）
  ```python
  # 同样的Bug，clean=pred_mag → clean=clean_mag
  ```

### 1.2 修复 `src/evaluation/metrics.py` STOI计算（严重）

- [ ] 替换 Pearson 相关系数为 `pystoi` 库的真正 STOI 算法
- [ ] 安装依赖：`pip install pystoi`

### 1.3 添加缺失指标到 `src/evaluation/metrics.py`（严重）

- [ ] 添加 `calculate_si_sdr` 方法（SI-SDR 指标）
- [ ] 添加 `calculate_pesq` 方法（PESQ 指标）
- [ ] 安装依赖：`pip install pesq`
- [ ] 在 `calculate_all_metrics` 中调用新指标

### 1.4 重构评估流程为时域评估（严重）

- [ ] 修改 `src/dataset.py` 的 `HowlingDataset`，额外返回：
  - STFT 复数矩阵（用于相位信息）
  - 或原始波形数据
- [ ] 修改 `src/evaluate.py` 的 `evaluate_with_metrics`：
  - 通过 iSTFT 将频谱还原为时域波形
  - 在时域信号上计算 STOI、PESQ
  - SI-SDR 可在频域或时域计算
- [ ] 修改 `src/evaluate.py` 的 `evaluate_traditional_methods` 同理

### 1.5 更新指标集合

- [ ] 更新 `calculate_all_metrics` 返回的指标为论文需要的：
  - `si_sdr_db`：SI-SDR（尺度不变信噪比）
  - `pesq_score`：PESQ（语音质量感知评估）
  - `stoi_score`：STOI（短时客观可懂度）
  - `snr_improvement_db`：SNR 改善量
  - `howling_reduction_db`：啸叫抑制量
- [ ] 更新 `calculate_mos_score` 使用新指标

---

## 第二阶段：实验重跑

### 实验1：5个U-Net模型训练（exp1_model_training）

- [ ] **模型训练**：无需重跑（训练过程不涉及评估Bug）
- [ ] **模型评估**：需要用修复后的代码重新评估全部5个模型
  - [ ] unet_v1 (AudioUNet3)
  - [ ] unet_v2 (AudioUNet5)
  - [ ] unet_v3_attention (AudioUNet5Attention)
  - [ ] unet_v6_optimized (AudioUNet5Optimized)
  - [ ] unet_v10_gan (AudioUNet5GAN)
- [ ] 更新 `organized_results/exp1_model_training/` 数据

### 实验2.2：传统方法评估（exp2_traditional_methods）

- [ ] 用修复后的代码重新评估3种传统方法
  - [ ] FrequencyShift（移频法）
  - [ ] GainSuppression（增益抑制法）
  - [ ] AdaptiveFeedback（自适应反馈消除法）
- [ ] 更新 `organized_results/exp2_traditional_methods/` 数据

### 实验3：统一评估（exp3_unified_evaluation）⭐ 核心实验

- [ ] 用修复后的代码重新运行统一评估（655个测试样本，9种方法）
- [ ] 更新 `organized_results/exp3_unified_evaluation/` 数据
- [ ] 更新 `experiment_results/evaluation_results.json`

### 实验4：损失函数对比（exp4_loss_comparison）

- [ ] 用修复后的代码重新评估不同损失函数训练的模型
  - [ ] MSE Loss
  - [ ] L1 Loss
  - [ ] SI-SDR Loss
  - [ ] MR-STFT Loss
  - [ ] Composite Loss
- [ ] 更新 `organized_results/exp4_loss_comparison/` 数据
- [ ] 更新论文表5-5数据

### 实验5：训练策略对比（exp5_training_strategy）

- [ ] 用修复后的代码重新评估不同训练策略的模型
  - [ ] Constant LR
  - [ ] StepLR
  - [ ] CosineAnnealing
  - [ ] WarmupCosine
  - [ ] ReduceLROnPlateau
- [ ] 更新 `organized_results/exp5_training_strategy/` 数据

### 实验6-7：数据增强对比（exp6_augmentation_comparison）

- [ ] 用修复后的代码重新评估不同增强策略训练的模型
  - [ ] Baseline（无增强）
  - [ ] SpecAugment
  - [ ] TimeMasking
  - [ ] FrequencyMasking
  - [ ] Combined
- [ ] 更新 `organized_results/exp6_augmentation_comparison/` 数据

---

## 第三阶段：论文数据更新

### 需要更新的论文章节和表格

- [ ] **第5章 实验4.1**：模型对比评估表（表5-1 或等效表格）
  - 5个U-Net模型的 SI-SDR、PESQ、STOI 数据
- [ ] **第5章 实验4.2**：传统方法对比表（表5-2 或等效表格）
  - 3种传统方法 vs 深度学习方法
- [ ] **第5章 实验4.3**：统一评估对比（核心表格）
  - 全部9种方法的综合对比
- [ ] **第5章 表5-5**：损失函数对比表
  - 5种损失函数的 SI-SDR、PESQ、STOI 数据
- [ ] **第5章 实验5**：训练策略对比表
  - 5种策略的训练效率和评估指标
- [ ] **第5章 实验6-7**：数据增强对比表
  - 5种增强策略的效果对比
- [ ] **第6章**：结果分析与讨论
  - 基于新数据重新撰写分析结论
- [ ] **摘要/结论**：更新核心数据

---

## 注意事项

1. **训练模型不需要重新训练**——训练过程使用的是 `train.py` 中的 SpectralLoss，不涉及评估Bug
2. **只需要重新运行评估**——用修复后的评估代码对已有模型重新打分
3. **预期变化**：
   - SNR 改善量将从 ~70dB 降到合理范围（预计 5~15dB）
   - STOI 将不再是 1.0（预计 0.7~0.95）
   - PESQ 将是新指标（预计 2.0~4.0）
   - SI-SDR 将是新指标（预计 5~15dB）
   - MOS 将从 4.6 降到合理范围
4. **实验优先级**：实验3（统一评估）> 实验4（损失函数）> 实验5（训练策略）> 实验6-7（增强对比）

---

## 快速命令参考

```bash
# 安装新依赖
pip install pystoi pesq

# 修复后重新评估（示例）
python -m src.evaluate --checkpoint <path> --full-metrics --compare-traditional --output-dir results/

# 运行完整实验脚本
python scripts/evaluate_all.py