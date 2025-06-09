# ===============[ IMPORTS ]===============
import torch


# ===============[ CORE ]===============
def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split images into non-overlapping patches.

    Parameters:
    - images: Input images of shape (B, C, H, W)
    - patch_size: Size of each square patch

    Returns:
    - Patches of shape (B, N, C * patch_size * patch_size) where N = (H/patch_size) * (W/patch_size)
    """
    B, C, H, W = images.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"Image dimensions {H}x{W} must be divisible by {patch_size}")

    # Reshape to (B, C, num_patches_y, patch_height, num_patches_x, patch_width)
    patches = images.view(B, C, H // patch_size, patch_size, W // patch_size, patch_size)

    # Rearrange to (B, num_patches_y, num_patches_x, C, P, P)
    patches = patches.permute(0, 2, 4, 1, 3, 5)

    # Reshape to (B, N, C * patch_size * patch_size)
    patches = patches.reshape(B, -1, C * patch_size * patch_size)

    return patches


def depatchify(patches: torch.Tensor, patch_size: int, image_size: int) -> torch.Tensor:
    """Convert patches back to images.

    Parameters:
    - patches: Patches of shape (B, N, C * patch_size * patch_size)
    - patch_size: Size of each square patch
    - image_size: Size of the output square image

    Returns:
    - Images of shape (B, C, image_size, image_size)
    """
    B, N, D = patches.shape
    C = D // (patch_size * patch_size)

    # Calculate grid dimensions
    grid_size = image_size // patch_size

    # Reshape to (B, grid_size, grid_size, C, patch_size, patch_size)
    patches = patches.reshape(B, grid_size, grid_size, C, patch_size, patch_size)

    # Permute to (B, C, grid_size, patch_size, grid_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5)

    # Reshape to (B, C, image_size, image_size)
    images = patches.reshape(B, C, image_size, image_size)

    return images


def get_num_patches(image_size: int, patch_size: int) -> int:
    """Get total number of patches in the image."""
    return (image_size // patch_size) * (image_size // patch_size)
