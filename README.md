<!--- Banner -->
<br />
<p align="center">
<a href="#"><img src="https://i.ibb.co/VcHWgmw4/image.png"></a>
<h3 align="center">Recurrent Interface Networks</h3>
<p align="center">Implementation of "Scalable Adaptive Computation for Iterative Generation" (ICML 2023)</p>

<!--- About --><br />
## About

This repository is an implementation of [Recurrent Interface Networks (RINs)](https://arxiv.org/abs/2212.11972), an attention-based architecture that decouples core computation from data dimensionality, enabling adaptive computation for scalable generation of high-dimensional data. RINs focus the bulk of computation on a set of latent tokens, using cross-attention to route information between latent and data tokens. This leads to state-of-the-art pixel diffusion models for image and video generation, scaling to 1024×1024 images without cascades or guidance, while being domain-agnostic and up to 10× more efficient than 2D and 3D U-Nets.

This implementation uses the RIN infrastructure in conjunction with a [Denoising Diffusion Probabilistic Model (DDPM)](https://arxiv.org/abs/2006.11239) for image generation.

<!--- Architecture --><br />
## Architecture
<a href="#"><img src="https://i.ibb.co/sYbRK10/image.png"></a>

The architecture consists of three key components:

- **Interface (X)**: Locally connected to input space and initialized via tokenization (e.g., patch embeddings). Grows linearly with input size.
- **Latents (Z)**: Decoupled from data, initialized as learnable embeddings. Much smaller than interface (e.g., hundreds vs thousands of vectors).
- **RIN Blocks**: Each block routes information between X and Z through three operations:
  - Read: Routes information from interface to latents via cross-attention
  - Compute: Processes information within latents via self-attention
  - Write: Projects information back to interface via cross-attention

The architecture uses latent self-conditioning to leverage recurrence in iterative generation tasks, allowing for propagation of routing context between iterations without backpropagation through time.

<!--- How it works --><br />
## How it works

RINs work by first tokenizing the input (e.g., images into patches) to form the interface X. A stack of RIN blocks then route information between X and a smaller set of latent tokens Z, avoiding quadratic pairwise interactions between tokens in X.

During training, the model learns to denoise images through a diffusion process. The recurrent interface mechanism helps maintain consistency across the generation process by:
1. Reading information from interface tokens into latents
2. Computing with self-attention within latents
3. Writing updates back to interface tokens
4. Using latent self-conditioning to propagate context between iterations

During inference, the model starts from random noise and iteratively denoises it while maintaining temporal consistency through the recurrent interface. This results in high-quality image generation with good coherence and detail preservation, while being significantly more efficient than traditional architectures.

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
    num_latents=64,
    num_blocks=2,
    block_depth=1,
    num_heads=4
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

<!--- License --><br />
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<!--- Citation --><br />
## Citation

If you use this code in your research, please cite both the original paper and this repository:

```bibtex
@inproceedings{jabri2023scalable,
  title={Scalable Adaptive Computation for Iterative Generation},
  author={Jabri, Allan and Fleet, David and Chen, Ting},
  booktitle={International Conference on Machine Learning},
  year={2023}
}

@software{rin_implementation,
  author = {Elian Belot},
  title = {RIN Implementation},
  year = {2024},
  url = {https://github.com/ElianBelot/rin}
}
```
