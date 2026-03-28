# Audio Feedback Suppression with U-Net

Graduation project focusing on audio feedback (howling) detection and elimination using deep learning.

## Quick Start

```bash
# Train baseline model
python src/train.py --config configs/unet_v2.yaml

# Train optimized model (recommended)
python src/train.py --config configs/unet_v6_optimized.yaml

# Evaluate model
python src/evaluate.py --checkpoint data/checkpoints/best_model.pth
```

## Project Structure

- `configs/` - YAML configuration files for all model variants
- `src/` - Source code
- `data/` - Training/validation data

## Documentation

See [configs/README.md](configs/README.md) for detailed configuration documentation.

## Requirements

- Python 3.8+
- PyTorch
- See `requirements.txt` for full dependencies

## License

© 2026 Graduation Project
