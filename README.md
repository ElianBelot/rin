<!--- Banner -->
<br />
<p align="center">
<a href="#"><img src="https://i.ibb.co/7pKLTjw/image.png"></a>
<h3 align="center">RIN</h3>
<p align="center">A recurrent interface network for high-quality image generation using diffusion models.</p>

<!--- About --><br />
## About

This project implements a diffusion-based image generation model using a recurrent interface network architecture. The model combines the strengths of diffusion models with a novel recurrent interface mechanism to generate high-quality images. This is a **work in progress**.

<!--- Architecture --><br />
## Architecture
<a href="#"><img src="https://i.ibb.co/7pKLTjw/image.png"></a>

The model consists of four main components:

- **RINModel**: The core recurrent interface network that processes image patches and maintains a latent state.
- **DiffusionModel**: Implements the diffusion process, handling the gradual denoising of images.
- **PatchProcessor**: Manages the conversion between image patches and latent representations.
- **InterfaceNetwork**: Handles the recurrent interface mechanism that allows for better temporal consistency in generation.

<!--- How it works --><br />
## How it works

RIN works by first breaking down images into patches, which are then processed through a recurrent interface network. The network maintains a latent state that evolves over time, allowing it to capture both local and global features of the image.

During training, the model learns to denoise images through a diffusion process, gradually transforming random noise into coherent images. The recurrent interface mechanism helps maintain consistency across the generation process, leading to higher quality results.

During inference, the model starts from random noise and iteratively denoises it while maintaining temporal consistency through the recurrent interface. This results in high-quality image generation with good coherence and detail preservation.

<!--- Installation --><br />
## Installation

```bash
# Clone the repository
git clone https://github.com/elian/rin.git
cd rin

# Install dependencies
pip install -e .
```

<!--- Usage --><br />
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

<!--- Development --><br />
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
