# ================[ IMPORTS ]================
import pytest
import torch

from src.utils.patch import depatchify, get_num_patches, patchify


# ================[ SETUP ]================
@pytest.fixture
def sample_image():
    """Create a sample image tensor for testing."""

    # (batch_size, channels, height, width)
    return torch.randn(2, 3, 64, 64)


# ================[ TESTS ]================
def test_patchify_valid_inputs(sample_image):
    """Test patchify with valid input dimensions."""
    # 64 patches (8x8 grid), each 192 values (3*8*8)
    patches = patchify(sample_image, patch_size=8)
    assert patches.shape == (2, 64, 192)

    # 256 patches (16x16 grid), each 48 values (3*4*4)
    patches = patchify(sample_image, patch_size=4)
    assert patches.shape == (2, 256, 48)


def test_patchify_invalid_inputs(sample_image):
    """Test patchify with invalid input dimensions."""
    with pytest.raises(ValueError):
        patchify(sample_image, patch_size=5)  # 64 is not divisible by 5

    with pytest.raises(ValueError):
        patchify(sample_image, patch_size=7)  # 64 is not divisible by 7


def test_patchify_content_preservation():
    """Test that patchify preserves image content correctly."""
    # # Set top-left 4x4 to 1.0 for all channels in sample image
    image = torch.zeros(1, 3, 8, 8)
    image[0, :, 0:4, 0:4] = 1.0

    # Patchify with 4x4 patches
    patches = patchify(image, patch_size=4)

    # Check that the first patch contains all 1.0s
    first_patch = patches[0, 0]
    assert torch.allclose(first_patch, torch.ones(48))

    # Check that the second patch contains all 0.0s
    second_patch = patches[0, 1]
    assert torch.allclose(second_patch, torch.zeros(48))


def test_depatchify_valid_inputs(sample_image):
    """Test depatchify with valid input dimensions."""
    # Test with 8x8 patches
    patches = patchify(sample_image, patch_size=8)
    reconstructed = depatchify(patches, patch_size=8, image_size=64)
    assert reconstructed.shape == (2, 3, 64, 64)
    assert torch.allclose(reconstructed, sample_image)

    # Test with 4x4 patches
    patches = patchify(sample_image, patch_size=4)
    reconstructed = depatchify(patches, patch_size=4, image_size=64)
    assert reconstructed.shape == (2, 3, 64, 64)
    assert torch.allclose(reconstructed, sample_image)


def test_depatchify_content_preservation():
    """Test that depatchify preserves patch content correctly."""
    # Create a simple test image with known values
    image = torch.zeros(1, 3, 8, 8)
    image[0, :, 0:4, 0:4] = 1.0  # Set top-left 4x4 to 1.0 for all channels

    # Patchify and depatchify
    patches = patchify(image, patch_size=4)
    reconstructed = depatchify(patches, patch_size=4, image_size=8)

    # Check that the reconstructed image matches the original
    assert torch.allclose(reconstructed, image)


def test_get_num_patches():
    """Test get_num_patches with various input sizes."""
    assert get_num_patches(64, 8) == 64  # 8x8 grid
    assert get_num_patches(64, 4) == 256  # 16x16 grid
    assert get_num_patches(32, 4) == 64  # 8x8 grid
    assert get_num_patches(16, 4) == 16  # 4x4 grid
