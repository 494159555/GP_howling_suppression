# 评估代码Bug诊断与修复方案

## 一、核心Bug：`evaluate_with_metrics` 中 `clean` 和 `enhanced` 参数传错

### 问题位置
`src/evaluate.py` 第 148-152 行：

```python
# ❌ 当前代码（BUG）
sample_metrics = metrics_calc.calculate_all_metrics(
    clean=pred_mag,      # 错误！把模型输出当作"干净参考"
    noisy=noisy_mag,
    enhanced=pred_mag,   # 错误！clean和enhanced是同一个东西
)
```

### Bug后果
| 指标 | 异常值 | 原因 |
|------|--------|------|
| SNR改善 | ~70dB（异常高） | clean=pred_mag, enhanced=pred_mag → 两者完全相同 → MSE=0 → SNR趋于无穷大 → 改善量=无穷大 - 有限值 ≈ 70dB |
| PSNR | Infinity | clean=pred_mag, enhanced=pred_mag → MSE=0 → 代码第97行直接return float('inf') |
| STOI | 全部1.0 | clean=pred_mag, enhanced=pred_mag → 两者相关系数=1.0 → stoi=(1+1)/2=1.0 |
| MOS | 全部4.6 | 因为STOI=1.0且PSNR=inf，加权后得到接近满分的MOS |

### 修复方案
```python
# ✅ 正确代码
sample_metrics = metrics_calc.calculate_all_metrics(
    clean=clean_mag,     # 正确：用真实的干净语音作为参考
    noisy=noisy_mag,     # 正确：带啸叫的输入
    enhanced=pred_mag,   # 正确：模型输出是"增强后"的信号
)
```

### 同样的Bug也存在于 `evaluate_traditional_methods`
位置：`src/evaluate.py` 第 232-235 行，修复方式相同。

---

## 二、次要Bug：`metrics.py` 中 STOI 计算方式不正确

### 问题位置
`src/evaluation/metrics.py` 第 109-138 行 `calculate_stoi` 方法：

```python
# ❌ 当前代码（简化的STOI，不准确）
correlation = np.corrcoef(clean_np, enhanced_np)[0, 1]
stoi_score = max(0, min(1, (correlation + 1) / 2))  # 把[-1,1]映射到[0,1]
```

### 问题
1. 使用Pearson相关系数代替STOI，这不是STOI的正确定义
2. `(correlation + 1) / 2` 的映射方式：当两信号完全相同时correlation=1 → stoi=1.0，这就是为什么即使修复了参数传递问题后STOI也可能偏高
3. 真正的STOI需要短时帧分析、1/3倍频程带分析等步骤

### 修复方案
```python
# ✅ 使用pystoi库计算真正的STOI
from pystoi import stoi as stoi_func

def calculate_stoi(self, clean: torch.Tensor, enhanced: torch.Tensor) -> float:
    clean_np = clean.detach().cpu().numpy()
    enhanced_np = enhanced.detach().cpu().numpy()
    
    if clean_np.ndim > 1:
        clean_np = clean_np.flatten()
        enhanced_np = enhanced_np.flatten()
    
    # 使用真正的STOI算法
    return float(stoi_func(clean_np, enhanced_np, self.sample_rate))
```

---

## 三、缺失指标：PESQ 和 SI-SDR 未实现

### 问题
论文中使用的三个核心指标是 **SI-SDR、PESQ、STOI**，但 `metrics.py` 中：
- ❌ 没有 SI-SDR（仅有 SNR improvement）
- ❌ 没有 PESQ
- ⚠️ STOI 使用了错误的简化算法

### 修复方案：添加 SI-SDR 和 PESQ

```python
def calculate_si_sdr(self, clean: torch.Tensor, enhanced: torch.Tensor) -> float:
    """计算尺度不变信噪比（SI-SDR）"""
    # 最优缩放投影
    s_target = (torch.sum(enhanced * clean, dim=-1) / 
                (torch.sum(clean * clean, dim=-1) + self.eps)) * clean
    # 失真分量
    e_noise = enhanced - s_target
    # SI-SDR
    si_sdr = 10 * torch.log10(
        torch.sum(s_target ** 2, dim=-1) / 
        (torch.sum(e_noise ** 2, dim=-1) + self.eps)
    )
    result = si_sdr.mean()
    return result.item() if hasattr(result, 'item') else float(result)

def calculate_pesq(self, clean: torch.Tensor, enhanced: torch.Tensor) -> float:
    """计算PESQ（语音质量感知评估）"""
    from pesq import pesq as pesq_func
    
    clean_np = clean.detach().cpu().numpy().flatten()
    enhanced_np = enhanced.detach().cpu().numpy().flatten()
    
    # 确保在[-1, 1]范围内
    clean_np = np.clip(clean_np, -1.0, 1.0)
    enhanced_np = np.clip(enhanced_np, -1.0, 1.0)
    
    return float(pesq_func(self.sample_rate, clean_np, enhanced_np, 'wb'))
```

---

## 四、评估流程的根本问题：在频谱域而非时域评估

### 问题
当前评估直接在**频谱幅度**上计算所有指标（SNR、STOI、PESQ等），但：
- **STOI** 和 **PESQ** 是为**时域波形**设计的，需要实际的音频信号
- 当前代码传入的是形状为 `[B, 1, 256, T]` 的频谱张量，而不是 `[B, 1, T]` 的波形
- 这意味着即使修复了参数传递，STOI和PESQ的计算结果仍然不准确

### 修复方案：评估流程需要 iSTFT 还原

```python
def evaluate_with_metrics(model, dataloader, device) -> Dict[str, float]:
    """修复后的评估流程"""
    from src.evaluation.metrics import AudioMetrics
    
    metrics_calc = AudioMetrics(sample_rate=cfg.SAMPLE_RATE)
    all_metrics = {'si_sdr': [], 'pesq': [], 'stoi': []}
    
    with torch.no_grad():
        for noisy_mag, clean_mag, noisy_stft, clean_stft in dataloader:
            # 1. 模型预测掩膜/频谱
            pred_mag = model(noisy_mag.to(device))
            
            # 2. ★ 关键：通过iSTFT还原为时域信号
            # 将预测幅度与原始相位结合
            enhanced_stft = pred_mag.cpu() * torch.exp(1j * noisy_stft.angle())
            enhanced_wav = torch.istft(enhanced_stft, n_fft=cfg.N_FFT, 
                                        hop_length=cfg.HOP_LENGTH)
            clean_wav = torch.istft(clean_stft, n_fft=cfg.N_FFT, 
                                     hop_length=cfg.HOP_LENGTH)
            noisy_wav = torch.istft(noisy_stft, n_fft=cfg.N_FFT, 
                                     hop_length=cfg.HOP_LENGTH)
            
            # 3. 在时域信号上计算指标
            sample_metrics = metrics_calc.calculate_all_metrics(
                clean=clean_wav,
                noisy=noisy_wav, 
                enhanced=enhanced_wav,
            )
```

**注意**：这需要数据加载器同时返回STFT复数矩阵（或相位信息），当前 `HowlingDataset` 可能需要修改。

---

## 五、修复优先级和步骤

### 第1步（最高优先级）：修复参数传递Bug
- 文件：`src/evaluate.py` 第148-152行、第232-235行
- 改动：`clean=pred_mag` → `clean=clean_mag`
- 影响：修复后SNR、PSNR、STOI不再是虚假的完美值

### 第2步（高优先级）：添加SI-SDR和PESQ指标
- 文件：`src/evaluation/metrics.py`
- 添加 `calculate_si_sdr` 和 `calculate_pesq` 方法
- 在 `calculate_all_metrics` 中调用

### 第3步（高优先级）：修复STOI计算
- 文件：`src/evaluation/metrics.py` 第109-138行
- 替换为 `pystoi` 库的正确实现

### 第4步（中优先级）：重构评估流程为时域评估
- 文件：`src/evaluate.py` 的 `evaluate_with_metrics` 函数
- 修改数据加载器以提供相位信息
- 通过iSTFT还原时域信号后再计算指标

### 第5步（低优先级）：重新运行全部实验
- 修复后重新训练和评估所有模型
- 用新的真实数据更新论文第5章和第6章

---

## 六、总结

| 问题 | 严重程度 | 影响范围 |
|------|---------|---------|
| clean/enhanced参数传反 | 🔴 致命 | 所有评估结果（SNR=70dB, STOI=1.0, PSNR=inf）全部无效 |
| STOI简化算法不正确 | 🟡 严重 | 即使修复参数，STOI值仍然不准确 |
| 缺少SI-SDR和PESQ | 🟡 严重 | 无法生成论文所需的核心指标 |
| 在频谱域计算时域指标 | 🟡 严重 | STOI/PESQ的计算语义不正确 |

**核心结论**：当前JSON中的实验结果全部不可信，需要按上述步骤修复后重新运行实验。