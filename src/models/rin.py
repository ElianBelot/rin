# ===============[ IMPORTS ]===============
import torch
import torch.nn as nn

from src.models.components.block import RINBlock
from src.models.components.latents import LatentInitializer
from src.models.components.time import TimeConditioner
from src.models.components.tokenizer import Tokenizer
from src.utils.patch import depatchify


# ===============[ CORE ]===============
class RINModel(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        latent_dim: int = 1024,
        interface_dim: int = 512,
        num_latents: int = 128,
        num_blocks: int = 6,
        block_depth: int = 4,
        num_heads: int = 8,
    ):
        """Complete Recurrent Interface Network (RIN) model.

        Parameters:
        - image_size: height/width of input images (assumed square)
        - patch_size: size of non-overlapping square patches
        - latent_dim: dimension of latent token embeddings
        - interface_dim: dimension of interface token embeddings
        - num_latents: number of latent tokens
        - num_blocks: number of RIN blocks stacked sequentially
        - block_depth: number of compute steps within each RIN block
        - num_heads: number of attention heads in attention layers
        """
        super().__init__()

        # Parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.interface_dim = interface_dim
        self.num_latents = num_latents
        self.num_blocks = num_blocks
        self.block_depth = block_depth
        self.num_heads = num_heads

        # Components
        self.tokenizer = Tokenizer(image_size=image_size, patch_size=patch_size, interface_dim=interface_dim)
        self.latent_initializer = LatentInitializer(num_latents=num_latents, latent_dim=latent_dim)
        self.time_conditioner = TimeConditioner(latent_dim=latent_dim)

        # Stack of RIN blocks
        self.blocks = nn.ModuleList(
            [
                RINBlock(latent_dim=latent_dim, interface_dim=interface_dim, num_heads=num_heads, depth=block_depth)
                for _ in range(num_blocks)
            ]
        )

        # Final readout
        self.readout = nn.Linear(interface_dim, 3 * patch_size * patch_size)
        self.layer_norm = nn.LayerNorm(interface_dim)

    def forward(
        self, xt: torch.Tensor, t: torch.Tensor, Z_prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns predicted noise given a noisy input image as well as the latent state Z'.

        Parameters:
        - xt: noisy input image (B, 3, H, W)
        - t: timestep tensor (B,)
        - Z_prev: optional previous latent state (B, N, D)

        Returns:
        - noise_pred: predicted noise image (B, 3, H, W)
        - Z: latent state Z' (B, N, D)
        """
        # [1] Tokenize input image
        X = self.tokenizer(xt)

        # Logging X stats
        logs = {}
        logs["X_mean"] = X.mean()
        logs["X_std"] = X.std()
        logs["X_norm"] = X.norm(p=2)

        # [2] Initialize latents with self-conditioning
        if Z_prev is not None:
            Z = self.latent_initializer.warm_start(Z_prev)
        else:
            Z = self.latent_initializer(batch_size=xt.size(0))

        # [3] Condition latents on timestep
        Z = self.time_conditioner(Z, t)

        # [4] Process through stacked RIN blocks
        for block in self.blocks:
            Z, X = block(Z, X)

        # [5] Linear readout: project interface tokens to patches
        patches_pred = self.readout(self.layer_norm(X))

        # [6] Reassemble image from patches
        noise_pred = depatchify(patches=patches_pred, patch_size=self.patch_size, image_size=self.image_size)

        return noise_pred, Z, logs
