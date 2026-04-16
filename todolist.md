# 毕业论文 Word 手动修改清单

> 以下内容基于 `thesis_md/` 源文件和 pandoc 转换结果分析，需在 `thesis_word/thesis_pandoc.docx` 中手动修改。

---

## D. 需插入图片的位置

### D1. 图2-1 声反馈正反馈过程示意图（必须）
- **位置**: 第2章 2.1.2节末尾，"> **[此处插入图2-1...]**" 引用块处
- **来源文件**: `02_chapter.md` 第 51 行
- **现状**: md源文件中已替换为文本占位符（含信号流程文字描述），pandoc 转换后为引用块文本
- **操作**: 删除 Word 中的占位文本，手动插入信号流程框图（用 PPT/Visio/draw.io 绘制：麦克风→放大器(G)→扬声器→声学路径h(t)→反馈信号d(t)→回到麦克风，形成闭环）

### D2-D4. 图4-7 典型样本频谱对比（必须，合并生成）
- **位置**: 第4章 4.5.1节
- **内容**: 7子图拼接：(a)干净语音 (b)带啸叫语音 (c)移频法 (d)自适应反馈消除法 (e)AudioUNet3 (f)AudioUNet5Attention (g)AudioUNet5Optimized
- **操作**: 在 Linux 服务器上运行以下命令生成：
  ```bash
  cd /path/to/GP_howling_suppression
  python scripts/generate_spectrograms.py --sample-idx 0 --output-dir experiments/spectrogram_comparison --individual
  ```
- **输出**: `experiments/spectrogram_comparison/figure4_7_spectrogram_comparison.png`（合并大图）+ 各方法单独频谱图
- **可选参数**: `--skip-models`（无checkpoint时仅生成传统方法）；`--sample-idx N`（选第N个测试样本）

### D5. 图4-X 模型性能演进可视化（建议）
- **位置**: 第4章 4.5.2节
- **来源文件**: `04_chapter.md` 第 158-160 行
- **现状**: 纯文字描述性能数据，无柱状图/折线图
- **操作**: （可选）插入各模型 SI-SDR/PESQ/STOI 对比柱状图

### D6. 图4-X 训练过程可视化（建议）
- **位置**: 第4章 4.5.3节
- **来源文件**: `04_chapter.md` 第 162-164 行
- **现状**: 纯文字描述训练曲线走势，无实际图片
- **操作**: （可选）插入训练损失/SI-SDR/PESQ 随 epoch 变化的曲线图


## F. 其他 Word 排版检查项

### F3. 参考文献格式
- **位置**: 第7章 参考文献
- **现状**: pandoc 已转换，30 条文献均存在
- **操作**: 检查格式是否符合学校要求（[J]期刊/[C]会议/[D]学位论文/[S]标准 等类型标注是否保留）


---
