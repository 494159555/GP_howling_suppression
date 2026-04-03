# GPU 利用率优化指南

> 本项目使用 8× NVIDIA GeForce RTX 3090 (24GB) 训练音频啸叫抑制模型。
> 初始 GPU 利用率仅 9%~34%，显存占用约 1.2GB/24GB，存在大量优化空间。

---

## 目录

- [问题诊断](#问题诊断)
- [优化措施总览](#优化措施总览)
- [🔴 优先级高 — 已实施](#优先级高--已实施)
  - [1. DDP 替代 DataParallel](#1-ddp-替代-dataparallel)
  - [2. 修复 DataLoader 预加载](#2-修复-dataloader-预加载)
- [🟠 优先级中 — 建议实施](#优先级中--建议实施)
  - [3. 增大 Batch Size](#3-增大-batch-size)
  - [4. torch.compile() 算子融合](#4-torchcompile-算子融合)
  - [5. cuDNN Benchmark](#5-cudnn-benchmark)
- [🟡 优先级低 — 按需实施](#优先级低--按需实施)
  - [6. 数据集 I/O 优化](#6-数据集-io-优化)
  - [7. 利用空闲 GPU](#7-利用空闲-gpu)
  - [8. 梯度累积](#8-梯度累积)
  - [9. 自动混合精度 (AMP)](#9-自动混合精度-amp)
  - [10. 模型并行 (Pipeline Parallelism)](#10-模型并行-pipeline-parallelism)
- [性能监控工具](#性能监控工具)
- [快速启动](#快速启动)

---

## 问题诊断

### 典型症状

```
+-----------------------------------------------------------------------------------------+
| GPU  Name                 Persistence-M | Memory-Usage  | GPU-Util  Compute M.       |
|=========================================+===============+==============================|
|   0  NVIDIA GeForce RTX 3090           |  1449MiB/24GB |     34%      Default        |
|   1  NVIDIA GeForce RTX 3090           |  1181MiB/24GB |     25%      Default        |
|   2  NVIDIA GeForce RTX 3090           |  1181MiB/24GB |      9%      Default        |
|   3  NVIDIA GeForce RTX 3090           |  1181MiB/24GB |     21%      Default        |
+-----------------------------------------------------------------------------------------+
```

### 根本原因

GPU 利用率低通常由以下原因导致：

| 原因 | 表现 | 诊断方法 |
|------|------|----------|
| **数据加载瓶颈** | GPU 频繁等待数据 | `gpu_util` 低、CPU 占用高 |
| **DataParallel 效率低** | 多卡扩展效率差 | GPU 0 负载远高于其他 |
| **Batch Size 太小** | 显存利用率 <50% | `nvidia-smi` 显存占用低 |
| **I/O 瓶颈** | 磁盘读取慢 | `iostat -x 1` 查看磁盘利用率 |
| **缺少算子优化** | 计算效率低 | 未使用 `cudnn.benchmark` |

---

## 优化措施总览

| # | 措施 | 预期提升 | 难度 | 状态 |
|---|------|----------|------|------|
| 1 | DDP 替代 DataParallel | +30~50% | 中等 | ✅ 已实施 |
| 2 | DataLoader 预加载优化 | 减少 CPU 瓶颈 | 简单 | ✅ 已实施 |
| 3 | 增大 Batch Size | 提升吞吐 | 简单 | 待实施 |
| 4 | torch.compile() | +10~30% | 简单 | 待实施 |
| 5 | cuDNN Benchmark | 小幅加速 | 一行 | ✅ 已实施 |
| 6 | 数据集 I/O 优化 | 减少 I/O 等待 | 视情况 | 待实施 |
| 7 | 利用空闲 GPU | 近线性加速 | 配合 DDP | 待实施 |
| 8 | 梯度累积 | 等效大 batch | 简单 | 待实施 |
| 9 | 自动混合精度 | 2x 显存节省 | 简单 | 已启用 |
| 10 | 模型并行 | 超大模型适用 | 复杂 | N/A |

---

## 🔴 优先级高 — 已实施

### 1. DDP 替代 DataParallel

#### 原理

| 特性 | DataParallel (DP) | DistributedDataParallel (DDP) |
|------|-------------------|-------------------------------|
| 进程模型 | 单进程多线程 | 多进程 |
| GIL 影响 | ❌ 受 Python GIL 限制 | ✅ 无 GIL 瓶颈 |
| 通信方式 | GPU 0 汇总再广播 | AllReduce 环形通信 |
| 扩展效率 | ~60% | ~90%+ |
| 内存平衡 | GPU 0 负载更高 | 各卡均衡 |

#### 实施方式

**代码改动（已完成）：**

```python
# ❌ 旧方式 (DataParallel)
if gpu_ids is not None and len(gpu_ids) > 1:
    model = nn.DataParallel(model, device_ids=gpu_ids)

# ✅ 新方式 (DDP)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 初始化
dist.init_process_group(backend='nccl')
local_rank = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')

# 模型包装
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 数据采样器
train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = DataLoader(dataset, sampler=train_sampler, ...)
```

**启动方式：**

```bash
# 4 卡训练
torchrun --nproc_per_node=4 src/train.py --model unet_v2

# 指定 GPU 0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 src/train.py

# 8 卡训练
torchrun --nproc_per_node=8 src/train.py --model unet_v6_optimized

# 配合配置文件
torchrun --nproc_per_node=4 src/train.py --config configs/unet_v2.yaml --batch-size 64
```

#### 关键注意事项

1. **Batch Size 语义变化**：`--batch-size 64` 表示每张卡 64 个样本，4 卡总 batch = 256
2. **学习率调整**：总 batch 增大时，通常需要相应增大学习率（线性缩放规则：`lr = base_lr × world_size`）
3. **只有主进程保存模型和写日志**，避免文件冲突
4. **每 epoch 调用 `sampler.set_epoch(epoch)`** 确保数据打乱

---

### 2. 修复 DataLoader 预加载

#### 原理

DataLoader 的 `persistent_workers` 和 `prefetch_factor` 两个参数对数据加载效率影响显著：

| 参数 | 作用 | 默认值 | 建议值 |
|------|------|--------|--------|
| `persistent_workers` | Worker 进程跨 epoch 复用 | `False` | `True` |
| `prefetch_factor` | 每个 worker 预取的 batch 数 | `2` | `4~8` |
| `pin_memory` | 使用页锁定内存加速 CPU→GPU 传输 | `False` | `True` |

- `persistent_workers=False` 时，每个 epoch 结束后 worker 进程被销毁，下个 epoch 重新创建，产生大量开销
- `prefetch_factor` 越大，GPU 空闲等待数据的概率越低

#### 实施方式（已完成）

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    num_workers=num_workers,
    pin_memory=True,
    sampler=train_sampler,
    persistent_workers=True if num_workers > 0 else False,  # ← 新增
    prefetch_factor=4 if num_workers > 0 else None,          # ← 新增
    drop_last=True,                                          # ← DDP 建议
)
```

#### num_workers 调优建议

```python
# 经验公式
num_workers = 4 * num_gpus  # 通常 4~8 倍 GPU 数量

# 单卡: 4~8 workers
# 4 卡: 16 workers
# 8 卡: 16~32 workers（注意 CPU 核心数限制）
```

---

## 🟠 优先级中 — 建议实施

### 3. 增大 Batch Size

#### 原理

当前模型约 4.5M 参数，每张 RTX 3090 仅占用 ~1.2GB 显存（总 24GB）。更大的 batch size 可以：
- 提高 GPU 并行度（GPU 擅长并行矩阵运算）
- 减少通信次数（DDP 中每 step 一次 AllReduce）
- 稳定梯度估计

#### 实施方式

```bash
# 方式1: 命令行参数
torchrun --nproc_per_node=4 src/train.py --batch-size 256

# 方式2: 修改配置文件
# configs/unet_v2.yaml
training:
  batch_size: 256

# 方式3: 修改默认值
# src/config.py
BATCH_SIZE = 256  # 每张卡的 batch size
```

#### Batch Size 调优参考

| GPU 数量 | 每卡 Batch | 总 Batch | 预计显存占用 |
|----------|-----------|----------|-------------|
| 1 | 64 | 64 | ~3 GB |
| 4 | 128 | 512 | ~6 GB |
| 4 | 256 | 1024 | ~12 GB |
| 8 | 256 | 2048 | ~12 GB |

> ⚠️ 增大 batch size 后需要相应调整学习率。线性缩放规则：`lr_new = lr_base × (total_batch / base_batch)`

---

### 4. torch.compile() 算子融合

#### 原理

PyTorch 2.0+ 引入的 `torch.compile()` 可以：
- **算子融合**：将多个小算子合并为一个 kernel，减少 GPU 全局内存访问
- **内存优化**：减少中间张量的内存分配
- **自动优化**：针对硬件生成最优执行计划

#### 实施方式

在 `train()` 函数中模型初始化后添加：

```python
import torch

model = model_class().to(device)

# PyTorch 2.0+ 算子编译优化
if hasattr(torch, 'compile'):
    model = torch.compile(model)
    print("🔥 torch.compile() 已启用")
```

#### 注意事项

- 首次编译较慢（几分钟），后续运行会使用缓存
- 建议在 `mode="reduce-overhead"` 下使用以获得最佳性能
- 与 DDP 兼容：先 compile 再 DDP 包装

```python
# 正确顺序
model = model_class().to(device)
model = torch.compile(model)  # 先编译
model = DDP(model, device_ids=[local_rank])  # 再 DDP
```

---

### 5. cuDNN Benchmark

#### 原理

当输入尺寸固定时，cuDNN 会在第一个 iteration 尝试多种卷积算法，选择最快的缓存下来。

#### 实施方式（已完成）

```python
# 在训练开始前添加
torch.backends.cudnn.benchmark = True
```

> ✅ 已在本次修改中添加到 `train()` 函数开头。

---

## 🟡 优先级低 — 按需实施

### 6. 数据集 I/O 优化

#### 问题分析

数据存储在 `/mnt/ent_disk0/syx/howling_data`（外部磁盘），可能存在 I/O 瓶颈。

#### 优化方案

**方案 A：预处理为张量缓存**

```python
# 将音频预处理为 .pt 文件，避免训练时重复计算 STFT
import torchaudio
import torch

def preprocess_to_tensor(wav_path, output_path):
    waveform, sr = torchaudio.load(wav_path)
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=512, hop_length=128, power=2.0
    )
    mag = spec_transform(waveform).sqrt()
    # 归一化...
    torch.save(mag, output_path)
```

**方案 B：使用 tmpfs（内存盘）**

```bash
# 创建内存盘
sudo mkdir -p /dev/shm/howling_data
sudo mount -t tmpfs -o size=20G tmpfs /dev/shm/howling_data

# 复制数据到内存盘
cp -r /mnt/ent_disk0/syx/howling_data/* /dev/shm/howling_data/

# 修改配置
# src/config.py
DATA_ROOT = Path("/dev/shm/howling_data")
```

**方案 C：检查磁盘性能**

```bash
# 测试磁盘读取速度
hdparm -Tt /dev/sdX

# 实时监控 I/O
iostat -x 1

# 查看是否是 I/O 瓶颈
pidstat -d 1 -p $(pgrep python)
```

---

### 7. 利用空闲 GPU

从 `nvidia-smi` 来看，GPU 5-7 完全空闲，GPU 4 被其他进程占用。

```bash
# 使用 GPU 0-3, 5-7（跳过 GPU 4）
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 torchrun --nproc_per_node=7 src/train.py

# 或只使用空闲的 GPU 5-7
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 src/train.py
```

---

### 8. 梯度累积

当显存不足以容纳更大的 batch size 时，可以通过梯度累积模拟大 batch：

```python
# 等效 batch_size = batch_per_gpu × world_size × accumulation_steps
accumulation_steps = 4

for i, (noisy, clean) in enumerate(dataloader):
    loss = criterion(model(noisy), clean) / accumulation_steps
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

---

### 9. 自动混合精度 (AMP)

> ✅ 已在项目中默认启用（`USE_MIXED_PRECISION = True`）

AMP 的优势：
- **显存节省**：FP16 仅需 FP32 一半的显存
- **加速计算**：RTX 3090 的 Tensor Core 对 FP16 运算更快
- **精度保持**：关键操作（如 loss scaling）仍使用 FP32

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

with autocast():
    pred = model(inputs)
    loss = criterion(pred, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

### 10. 模型并行 (Pipeline Parallelism)

当模型太大无法放入单张 GPU 时使用。对于本项目的轻量模型（~4.5M），不需要模型并行。

但如果未来使用更大的模型（如 Transformer-based），可以考虑：

```python
# 简单示例：将模型不同层放在不同 GPU
model.enc1 = model.enc1.to('cuda:0')
model.enc2 = model.enc2.to('cuda:0')
model.dec1 = model.dec1.to('cuda:1')
model.dec2 = model.dec2.to('cuda:1')
```

---

## 性能监控工具

### 实时 GPU 监控

```bash
# 方式1: nvidia-smi 持续监控（每1秒刷新）
watch -n 1 nvidia-smi

# 方式2: 更详细的 GPU 指标
nvidia-smi dmon -s pucvmet -i 0,1,2,3 -d 1

# 方式3: gpustat（需安装）
pip install gpustat
gpustat -i 1
```

### PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train_one_epoch(model, dataloader, ...)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 分析 GPU 利用率瓶颈

```bash
# 查看 CPU 是否是瓶颈
htop  # 如果 DataLoader 进程的 CPU 接近 100%，说明数据加载慢

# 查看磁盘 I/O
iostat -x 1

# 查看进程级的 GPU 使用
nvidia-smi pmon -c 5
```

---

## 快速启动

### 单卡训练

```bash
python src/train.py --model unet_v2 --batch-size 128
```

### 4 卡 DDP 训练（推荐）

```bash
torchrun --nproc_per_node=4 src/train.py \
    --model unet_v2 \
    --batch-size 128 \
    --lr 4e-4 \
    --mixed-precision
```

### 8 卡 DDP 训练（跳过被占用的 GPU 4）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 torchrun --nproc_per_node=7 src/train.py \
    --model unet_v6_optimized \
    --batch-size 256 \
    --lr 8e-4 \
    --loss multitask \
    --mixed-precision
```

### 调试模式

```bash
torchrun --nproc_per_node=2 src/train.py --debug
```

---

## 性能优化检查清单

- [x] 使用 DDP 替代 DataParallel
- [x] 启用 DataLoader persistent_workers
- [x] 设置 prefetch_factor=4
- [x] 启用 pin_memory=True
- [x] 启用 cuDNN benchmark
- [x] 已启用混合精度训练 (AMP)
- [ ] 增大 batch size（根据显存情况调整）
- [ ] 启用 torch.compile()
- [ ] 检查数据 I/O 是否为瓶颈
- [ ] 使用更多空闲 GPU
- [ ] 调整 num_workers（建议 4×GPU数）

---

## 参考资料

- [PyTorch Distributed Data Parallel 官方教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [torch.compile 文档](https://pytorch.org/docs/stable/generated/torch.compile.html)
- [NVIDIA DPU Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
+++++++ REPLACE