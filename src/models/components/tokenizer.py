import torch
import torch.nn as nn

from src.utils.patch import get_num_patches, patchify


class Tokenizer(nn.Module):

    def __init__(self, image_size: int, patch_size: int, interface_dim: int):
        """Converts an input image into a sequence of patch tokens with positional encodings.

        Parameters:
        - image_size: height and width of input images
        - patch_size: height and width of each square patch
        - interface_dim: dimension of the embedding space for each patch
        """
        super().__init__()

        self.patch_size = patch_size
        self.interface_dim = interface_dim
        self.image_size = image_size

        # Project each flattened patch to a token
        num_patches = get_num_patches(image_size, patch_size)
        self.linear = nn.Linear(patch_size * patch_size * 3, interface_dim)
        self.layer_norm = nn.LayerNorm(interface_dim)

        # Learnable positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, interface_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify image, flatten patches, project to token space, and add positional encodings.

        Parameters:
        - x: input image tensor of shape (B, 3, H, W)

        Returns:
        - Patch tokens of shape (B, N, dim)
        """
        B, C, H, W = x.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")
        if H != self.image_size or W != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}x{self.image_size}, got {H}x{W}")

        patches = patchify(x, self.patch_size)
        tokens = self.linear(patches)
        tokens = self.layer_norm(tokens) + self.pos_embed

        return tokens
