# AGENTS.md

## Cursor Cloud specific instructions

### 环境概述

本项目为基于 PyTorch 的音频啸叫抑制深度学习项目，无外部服务依赖（无数据库、无 Web 服务）。所有工作流通过 CLI Python 脚本完成。

### 关键注意事项

- **PyTorch 版本**：必须使用 PyTorch 2.2.x（CPU-only），因为 `src/train.py` 使用了 `ReduceLROnPlateau(verbose=True)`，该参数在 PyTorch 2.3+ 中被移除。安装命令：`pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu`
- **无 GPU**：Cloud VM 没有 GPU，训练在 CPU 上运行。使用 `--no-amp` 避免 AMP 警告，使用 `--debug` 模式快速测试（3 epochs, batch_size=2）
- **PYTHONPATH**：运行所有脚本时需设置 `PYTHONPATH=/workspace`，否则 `from src.xxx` 导入会失败
- **训练数据**：代码硬编码数据路径为 `/mnt/ent_disk0/syx/howling_data`（见 `src/config.py`）。Cloud VM 上需要先创建该目录并生成合成数据（或挂载真实数据）才能运行训练/评估
- **无 lint/test 框架**：项目未配置 pylint/flake8/pytest 等工具，无自动化测试套件

### 运行命令

参考 `CLAUDE.md` 中的 Commands 部分。在 Cloud VM 上运行时加上环境前缀：

```bash
# 快速验证训练流程
PYTHONPATH=/workspace python3 src/train.py --config configs/unet_v2.yaml --debug --no-amp --num-workers 0

# 评估模型
PYTHONPATH=/workspace python3 src/evaluate.py --checkpoint <path_to_pth> --batch-size 2
```
