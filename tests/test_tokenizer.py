# ================[ IMPORTS ]================
import pytest
import torch

from src.models.tokenizer import Tokenizer
from src.utils.patch import patchify


# ================[ SETUP ]================
@pytest.fixture
def tokenizer():
    """Create a tokenizer instance for testing."""
    return Tokenizer(patch_size=8, dim=256, image_size=64)


@pytest.fixture
def small_tokenizer():
    """Create a tokenizer instance for testing with smaller image size."""
    return Tokenizer(patch_size=4, dim=128, image_size=32)


# ================[ TESTS ]================
def test_tokenizer_initialization():
    """Test tokenizer initialization with different parameters."""
    # Test with default image size
    tokenizer = Tokenizer(patch_size=8, dim=256)
    assert tokenizer.patch_size == 8
    assert tokenizer.dim == 256
    assert tokenizer.image_size == 64

    # Test with custom image size
    tokenizer = Tokenizer(patch_size=4, dim=128, image_size=32)
    assert tokenizer.patch_size == 4
    assert tokenizer.dim == 128
    assert tokenizer.image_size == 32


def test_tokenizer_forward(tokenizer):
    """Test tokenizer forward pass."""
    # (batch_size, channels, height, width)
    x = torch.randn(2, 3, 64, 64)

    # Forward pass
    tokens = tokenizer(x)

    # Check output shape
    expected_patches = (64 // 8) * (64 // 8)  # 8x8 grid
    assert tokens.shape == (2, expected_patches, 256)

    # Check that positional embeddings are added
    patches = patchify(x, tokenizer.patch_size)
    linear_output = tokenizer.linear(patches)
    assert not torch.allclose(tokens, linear_output)


def test_tokenizer_different_batch_sizes(tokenizer):
    """Test tokenizer with different batch sizes."""
    # Test with batch size 1
    x1 = torch.randn(1, 3, 64, 64)
    tokens1 = tokenizer(x1)
    assert tokens1.shape == (1, 64, 256)

    # Test with batch size 4
    x4 = torch.randn(4, 3, 64, 64)
    tokens4 = tokenizer(x4)
    assert tokens4.shape == (4, 64, 256)


def test_tokenizer_invalid_input(small_tokenizer):
    """Test tokenizer with invalid input dimensions."""
    # Test with wrong image size
    x = torch.randn(2, 3, 64, 64)  # 64x64 instead of 32x32
    with pytest.raises(ValueError):
        small_tokenizer(x)

    # Test with wrong number of channels
    x = torch.randn(2, 1, 32, 32)  # 1 channel instead of 3
    with pytest.raises(ValueError):
        small_tokenizer(x)
