# 论文相关文件结构

本文档整理了项目中与毕业论文直接相关的文件及其用途说明。

---

## 📁 thesis/ — 论文章节（Markdown源文件）

| 文件 | 对应章节 | 说明 |
|------|---------|------|
| `00_abstract.md` | 摘要 | 中英文摘要及关键词 |
| `01_chapter1.md` | 第1章 绪论 | 研究背景与意义、国内外研究现状、研究内容与结构安排 |
| `02_chapter2.md` | 第2章 相关理论基础 | 声学啸叫模型、U-Net架构、注意力机制、残差连接与空洞卷积、STFT |
| `02_chapter2_word_formulas.md` | 第2章公式辅助 | Word公式格式的对照版本，方便在Word中插入公式 |
| `03_chapter3.md` | 第3章 实验设计 | 数据集构建、特征提取、四个U-Net模型架构、传统算法、损失函数、数据增强、评价指标 |
| `04_chapter4.md` | 第4章 实验组织 | 实验环境、四组实验方案（模型对比、统一评估、消融研究、训练优化） |
| `05_chapter5.md` | 第5章 实验结果与分析 | 基线结果、模型演进对比、消融分析、训练优化结果、可视化、与Deep AHS对比 |
| `06_chapter6.md` | 第6章 结语 | 工作总结、主要结论、不足与展望 |
| `07_acknowledgements.md` | 致谢 | 致谢词 |

---

## 📁 thesis_word/ — 论文章节（Word文档）

| 文件 | 说明 |
|------|------|
| `00_abstract.docx` | 摘要 Word版 |
| `01_chapter1.docx` | 第1章 Word版 |
| `02_chapter2.docx` | 第2章 Word版 |
| `03_chapter3.docx` | 第3章 Word版 |
| `04_chapter4.docx` | 第4章 Word版 |
| `05_chapter5.docx` | 第5章 Word版 |
| `06_chapter6.docx` | 第6章 Word版 |
| `07_acknowledgements.docx` | 致谢 Word版 |

---

## 📁 configs/ — 模型配置文件

| 文件 | 对应论文内容 | 说明 |
|------|-------------|------|
| `unet_v1.yaml` | AudioUNet3（第3章 3.3.1） | 3层基线U-Net配置 |
| `unet_v2.yaml` | AudioUNet5（第3章 3.3.2） | 5层标准U-Net配置 |
| `unet_v3_attention.yaml` | AudioUNet5Attention（第3章 3.3.3） | 注意力U-Net配置 |
| `unet_v6_optimized.yaml` | AudioUNet5Optimized（第3章 3.3.4） | 综合优化U-Net配置 |

---

## 📁 src/ — 源代码

### src/ 根目录

| 文件 | 说明 |
|------|------|
| `config.py` | 配置管理，加载YAML配置文件 |
| `dataset.py` | 数据集类，加载和处理STFT特征数据 |
| `evaluate.py` | 模型评估入口 |
| `train.py` | 模型训练入口 |
| `train_v2.py` | 模型训练入口（改进版） |

### src/models/ — 模型定义

| 文件 | 对应论文内容 | 说明 |
|------|-------------|------|
| `unet_v1.py` | 第3章 3.3.1 | AudioUNet3 三层基线U-Net |
| `unet_v2.py` | 第3章 3.3.2 | AudioUNet5 五层标准U-Net |
| `unet_v3_attention.py` | 第3章 3.3.3 | AudioUNet5Attention 注意力U-Net |
| `unet_v6_optimized.py` | 第3章 3.3.4 | AudioUNet5Optimized 综合优化U-Net |
| `blocks.py` | 第3章 3.3 | 通用网络组件（残差块、空洞卷积块等） |
| `modules/attention_modules.py` | 第2章 2.3 | 注意力门模块实现 |
| `loss_functions.py` | 第3章 3.5 | 损失函数（MSE、L1、频谱损失、组合损失等） |
| `augmentation.py` | 第3章 3.6 | 数据增强（SpecAugment、Mixup等） |
| `training_strategies.py` | 第4章 4.2.4 | 学习率调度策略（CosineAnnealing、Warmup+CosineDecay等） |
| `post_processing.py` | — | 后处理工具 |

### src/traditional/ — 传统算法实现

| 文件 | 对应论文内容 | 说明 |
|------|-------------|------|
| `frequency_shift.py` | 第3章 3.4.1 | 移频法 |
| `gain_suppression.py` | 第3章 3.4.2 | 增益抑制法 |
| `adaptive_feedback.py` | 第3章 3.4.3 | 自适应反馈消除法（NLMS） |

### src/evaluation/ — 评估工具

| 文件 | 对应论文内容 | 说明 |
|------|-------------|------|
| `metrics.py` | 第3章 3.7 | 评价指标计算（SI-SDR、PESQ、STOI） |
| `comparator.py` | 第5章 | 多方法对比评估 |
| `visualizer.py` | 第4章 4.3 | 结果可视化（频谱图、训练曲线、雷达图等） |

---

## 📁 scripts/ — 实验脚本

| 文件 | 对应论文内容 | 说明 |
|------|-------------|------|
| `train_all_models.py` | 第4章 4.2.1 | 实验1：训练所有模型变体 |
| `evaluate_all.py` | 第4章 4.2.2 | 实验2：统一性能评估 |
| `ablation_study.py` | 第4章 4.2.3 | 实验3：消融实验 |
| `loss_comparison.py` | 第4章 4.2.4 | 实验4：损失函数对比 |
| `training_strategy_comparison.py` | 第4章 4.2.4 | 实验4：学习率调度策略对比 |
| `exp5_training_strategy.py` | 第4章 4.2.4 | 实验4：训练策略实验辅助脚本 |
| `augmentation_comparison.py` | 第4章 4.2.4 | 实验4：数据增强策略对比 |
| `test_traditional_methods.py` | 第4章 4.2.2 | 传统方法测试脚本 |
| `generate_report.py` | 第5章 | 实验报告生成脚本 |

---

## 📁 experiment_results/ — 实验结果数据

| 路径 | 说明 |
|------|------|
| `evaluation_results.json` | 统一评估结果汇总 |
| `generate_final_report.py` | 最终报告生成脚本 |
| `exp2_2_traditional/` | 传统方法评估结果 |
| `exp3_unified_eval/` | 统一评估详细结果 |

---

## 📁 organized_results/ — 整理后的实验结果

| 路径 | 对应论文内容 | 说明 |
|------|-------------|------|
| `exp1_model_training/` | 第5章 5.2 | 模型训练过程数据与曲线 |
| `exp2_traditional_methods/` | 第5章 5.1.2 | 传统方法基线结果 |
| `exp3_unified_evaluation/` | 第5章 5.2 | 统一评估对比数据 |
| `exp4_loss_comparison/` | 第5章 5.4.1 | 损失函数对比结果 |
| `exp5_training_strategy/` | 第5章 5.4.2 | 学习率调度策略对比结果 |
| `exp6_augmentation_comparison/` | 第5章 5.4.3 | 数据增强策略对比结果 |

---

## 🔧 工具脚本

| 文件 | 说明 |
|------|------|
| `convert_to_word.py` | 将 thesis/ 下的 Markdown 文件批量转换为 Word 文档 |