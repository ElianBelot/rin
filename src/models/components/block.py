# ================[ IMPORTS ]================
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


# ================[ CORE ]================
class RINBlock(nn.Module):

    def __init__(self, latent_dim: int, interface_dim: int, num_heads: int = 8, depth: int = 1):
        """Recurrent Interface Network (RIN) block."""
        super().__init__()

        # [1] Read (X -> Z)
        self.read_attention = MultiheadAttention(
            embed_dim=latent_dim, kdim=interface_dim, vdim=interface_dim, num_heads=num_heads, batch_first=True
        )
        self.read_mlp = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )

        # [2] Compute (Z -> Z)
        self.compute_attention = nn.ModuleList(
            [MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True) for _ in range(depth)]
        )
        self.compute_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.Linear(latent_dim, latent_dim * 4),
                    nn.GELU(),
                    nn.Linear(latent_dim * 4, latent_dim),
                )
                for _ in range(depth)
            ]
        )

        # [3] Write (Z -> X)
        self.write_attention = MultiheadAttention(
            embed_dim=interface_dim, kdim=latent_dim, vdim=latent_dim, num_heads=num_heads, batch_first=True
        )
        self.write_mlp = nn.Sequential(
            nn.LayerNorm(interface_dim),
            nn.Linear(interface_dim, interface_dim * 4),
            nn.GELU(),
            nn.Linear(interface_dim * 4, interface_dim),
        )

        # Norms
        self.read_norm = nn.LayerNorm(latent_dim)
        self.compute_norm = nn.ModuleList([nn.LayerNorm(latent_dim) for _ in range(depth)])
        self.write_norm = nn.LayerNorm(interface_dim)

    def forward(self, Z: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through a RIN block.

        Parameters:
        - Z: Latent tokens (B, N_latents, D_latent)
        - X: Interface tokens (B, N_patches, D_interface)

        Returns:
        - Z: Updated latent tokens (B, N_latents, D_latent)
        - X: Updated interface tokens (B, N_patches, D_interface)
        """
        # [1] Read (X -> Z)
        Z = Z + self.read_attention(query=self.read_norm(Z), key=X, value=X)[0]
        Z = Z + self.read_mlp(Z)

        # [2] Compute (Z -> Z)
        for norm, MHA, MLP in zip(self.compute_norm, self.compute_attention, self.compute_mlp):
            Zn = norm(Z)
            Z = Z + MHA(query=Zn, key=Zn, value=Zn)[0]
            Z = Z + MLP(Z)

        # [3] Write (Z -> X)
        X = X + self.write_attention(query=self.write_norm(X), key=Z, value=Z)[0]
        X = X + self.write_mlp(X)

        return Z, X
