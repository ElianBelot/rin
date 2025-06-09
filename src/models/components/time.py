# ===============[ IMPORTS ]===============
import torch
import torch.nn as nn


# ===============[ CORE ]===============
class TimeConditioner(nn.Module):
    def __init__(self, latent_dim: int, time_dim: int = 128):
        """Time embedding + projection into latent space."""
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, latent_dim),
        )

    def forward(self, Z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add timestep conditioning to latent tokens."""
        t = t.view(-1, 1)
        time_embed = self.time_mlp(t)[:, None, :]
        return Z + time_embed
