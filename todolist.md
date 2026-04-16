# 毕业论文 Word 手动修改清单

> 以下内容基于 `thesis_md/` 源文件和 pandoc 转换结果分析，需在 `thesis_word/thesis_pandoc.docx` 中手动修改。

---

## D. 需插入图片的位置

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

---
