# CLAUDE.md — GP_howling_suppression

Audio feedback (howling) detection and suppression using deep learning (U-Net variants). Graduation project at Sichuan University.

## Environment

- **Python 3.10**, PyTorch, managed via conda (`environment.yml`)
- Conda env name: `GraduationProject`
- Data lives at `/mnt/ent_disk0/syx/howling_data` (train/dev splits with clean/howling subdirs)
- `experiments/` and `data/` are gitignored — output only

## Project Structure

```
configs/          # YAML configs per model variant (unet_v1..v10_gan)
src/
  config.py       # Global Config class (paths, hyperparams, model registry)
  dataset.py      # Audio dataset loader
  train.py        # Training entry point (uses YAML config)
  train_v2.py     # Alternative trainer
  evaluate.py     # Evaluation entry point
  models/         # Model definitions (unet_v1, v2, v3_attention, v6_optimized, v10_gan)
  models/blocks.py, modules/   # Shared building blocks
  models/loss_functions.py     # L1, MSE, spectral, multitask, adversarial losses
  models/augmentation.py       # Data augmentation
  models/post_processing.py    # Adaptive, smoothing, gain control post-processing
  models/training_strategies.py
  evaluation/     # metrics.py, comparator.py, visualizer.py
  traditional/    # Baseline methods: adaptive_feedback, frequency_shift, gain_suppression
scripts/          # Batch scripts: train_all_models, evaluate_all, ablation_study, etc.
experiments/      # Experiment output dirs (gitignored)
thesis/           # Thesis materials
```

## Commands

```bash
# Train (via YAML config)
python src/train.py --config configs/<variant>.yaml

# Evaluate
python src/evaluate.py --checkpoint <path_to_pth>

# Batch train all models
python scripts/train_all_models.py

# Batch evaluate all models
python scripts/evaluate_all.py
```

## Model Variants

| Config | Model Class | Description |
|--------|------------|-------------|
| unet_v1 | AudioUNet3 | 3-layer U-Net (lightweight baseline) |
| unet_v2 | AudioUNet5 | 5-layer U-Net (default) |
| unet_v3_attention | AudioUNet5Attention | 5-layer + attention gates |
| unet_v6_optimized | AudioUNet5Optimized | 5-layer + attention + residual + dilated |
| unet_v10_gan | AudioUNet5GAN | 5-layer + GAN framework |

## Audio Parameters

- Sample rate: 16000 Hz, chunk length: 3s
- STFT: N_FFT=512, hop_length=128

## Notes

- Comments and docs are in Chinese — keep new comments/docs in Chinese to match the project style
- GPU training expected (mixed precision / AMP enabled by default)
- Default loss: multitask; default model: unet_v2
