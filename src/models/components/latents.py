# ================[ IMPORTS ]================
import torch
import torch.nn as nn


# ================[ CORE ]================
class LatentInitializer(nn.Module):
    def __init__(self, num_latents: int, latent_dim: int):
        super().__init__()
        self.Z_init = nn.Parameter(torch.randn(1, num_latents, latent_dim))

        # LayerNorm with zero scaling and bias
        self.layer_norm = nn.LayerNorm(latent_dim, elementwise_affine=True)
        nn.init.zeros_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

        # MLP to warm-start latents
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """Initialize latents from scratch."""
        Z = self.Z_init.expand(batch_size, -1, -1)
        return Z

    def warm_start(self, Z_prev: torch.Tensor) -> torch.Tensor:
        """Warm-start latents from previous latents."""
        batch_size = Z_prev.size(0)

        Z_init = self.forward(batch_size)
        Z = Z_init + self.layer_norm(Z_prev + self.mlp(Z_prev))
        return Z
