# 实验重跑清单

> 基于 `evaluation_fix_plan.md` 中诊断的Bug，当前所有评估数据不可信
> 创建日期：2026-04-12

---

## 实验重跑

### 实验1：4个U-Net模型训练（exp1_model_training）

- [ ] **模型训练**：无需重跑（训练过程不涉及评估Bug）
- [ ] **模型评估**：需要用修复后的代码重新评估全部4个模型
  - [ ] unet_v1 (AudioUNet3)
  - [ ] unet_v2 (AudioUNet5)
  - [ ] unet_v3_attention (AudioUNet5Attention)
  - [ ] unet_v6_optimized (AudioUNet5Optimized)
- [ ] 更新 `experiments/exp1_model_training/` 数据

### 实验2.2：传统方法评估（包含在exp2统一评估中）

- [x] 已在 exp2 统一评估中完成3种传统方法评估
  - FrequencyShift（移频法）
  - GainSuppression（增益抑制法）
  - AdaptiveFeedback（自适应反馈消除法）

### 实验2：统一评估（exp2_unified_evaluation）⭐ 核心实验

- [ ] 用修复后的代码重新运行统一评估（655个测试样本，7种方法）
- [ ] 更新 `experiments/exp2_unified_evaluation/` 数据
- [ ] 更新 `experiments/experiment_results/evaluation_results.json`

### 实验4：损失函数对比（exp4_loss_comparison）

- [ ] 用修复后的代码重新评估不同损失函数训练的模型
  - [ ] MSE Loss
  - [ ] L1 Loss
  - [ ] SI-SDR Loss
  - [ ] MR-STFT Loss
  - [ ] Composite Loss
- [ ] 更新 `experiments/exp4_loss_comparison/` 数据
- [ ] 更新论文表5-5数据

### 实验5：训练策略对比（exp5_training_strategy）

- [ ] 用修复后的代码重新评估不同训练策略的模型
  - [ ] Constant LR
  - [ ] StepLR
  - [ ] CosineAnnealing
  - [ ] WarmupCosine
  - [ ] ReduceLROnPlateau
- [ ] 更新 `experiments/exp5_training_strategy/` 数据

### 实验6-7：数据增强对比（exp6_augmentation_comparison）

- [ ] 用修复后的代码重新评估不同增强策略训练的模型
  - [ ] Baseline（无增强）
  - [ ] SpecAugment
  - [ ] TimeMasking
  - [ ] FrequencyMasking
  - [ ] Combined
- [ ] 更新 `experiments/exp6_augmentation_comparison/` 数据

---

## 第三阶段：论文数据更新

### 需要更新的论文章节和表格

- [ ] **第5章 实验4.1**：模型对比评估表（表5-1 或等效表格）
  - 4个U-Net模型的 SI-SDR、PESQ、STOI 数据
- [ ] **第5章 实验4.2**：传统方法对比表（表5-2 或等效表格）
  - 3种传统方法 vs 深度学习方法
- [ ] **第5章 实验4.3**：统一评估对比（核心表格）
  - 全部8种方法的综合对比
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