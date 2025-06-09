# RIN: Recurrent Interface Network for Image Generation

A PyTorch implementation of a diffusion-based image generation model using a recurrent interface network architecture. This project provides tools for training and sampling from the model, with a focus on high-quality image generation.

## Features

- Diffusion-based image generation
- Recurrent interface network architecture
- Support for CelebA dataset
- PyTorch Lightning integration
- Weights & Biases logging
- Comprehensive test suite

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rin.git
cd rin

# Install dependencies
pip install -e .
```

## Usage

### Training

```python
from src.data.celeba import CelebADataModule
from src.models.rin import RINModel
from src.models.diffusion import DiffusionModel

# Initialize data module
datamodule = CelebADataModule(
    data_dir="./data",
    image_size=64,
    batch_size=64
)

# Create model
model = RINModel(
    image_size=64,
    patch_size=8,
    latent_dim=256,
    interface_dim=128,
    num_latents=64
)

# Train
trainer.fit(model, datamodule=datamodule)
```

### Sampling

```python
from src.models.rin import RINModel
from src.models.diffusion import DiffusionModel

# Load trained model
model = RINModel.load_from_checkpoint("path/to/checkpoint.ckpt")

# Generate samples
samples = model.sample(num_samples=4)
```

## Development

### Setup

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

### Testing

Run the test suite:
```bash
pytest
```
