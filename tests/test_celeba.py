# ================[ IMPORTS ]================
import os

import pytest
import torch

from src.data.celeba import CelebADataModule


# ================[ FIXTURES ]================
@pytest.fixture
def data_module(data_path: str = "./data"):
    """Create a CelebADataModule instance for testing."""
    return CelebADataModule(data_dir=data_path, image_size=64, batch_size=4, num_workers=0)


# ================[ TESTS ]================
def test_data_module_initialization(data_module):
    """Test data module initialization."""
    assert data_module.image_size == 64
    assert data_module.batch_size == 4
    assert data_module.num_workers == 0


def test_data_module_prepare_data(data_module):
    """Test data preparation."""
    # This will download the dataset if not present
    data_module.prepare_data()

    # Check that the data directory exists
    assert os.path.exists(data_module.data_dir)


def test_data_module_setup(data_module):
    """Test data module setup."""
    data_module.setup()

    # Check that dataset is created
    assert data_module.dataset is not None

    # Check dataset size
    assert len(data_module.dataset) > 0

    # Check that transform is applied correctly
    sample_image, _ = data_module.dataset[0]
    assert sample_image.shape == (3, 64, 64)
    assert torch.all(sample_image >= -1) and torch.all(sample_image <= 1)  # Check normalization


def test_data_module_dataloader(data_module):
    """Test dataloader creation and batch properties."""
    data_module.setup()
    dataloader = data_module.train_dataloader()

    # Check that dataloader is created
    assert dataloader is not None

    # Get a batch
    batch = next(iter(dataloader))
    images, labels = batch

    # Check batch properties
    assert images.shape == (4, 3, 64, 64)  # (batch_size, channels, height, width)
    assert torch.all(images >= -1) and torch.all(images <= 1)  # Check normalization
    assert labels.shape[0] == 4  # Check labels batch size


def test_data_module_transforms(data_module):
    """Test that transforms are applied correctly."""
    data_module.setup()

    # Get a sample image
    image, _ = data_module.dataset[0]

    # Check image properties
    assert image.shape == (3, 64, 64)
    assert torch.all(image >= -1) and torch.all(image <= 1)  # Check normalization
    assert image.dtype == torch.float32  # Check dtype


def test_data_module_multiple_batches(data_module):
    """Test that multiple batches can be loaded."""
    data_module.setup()
    dataloader = data_module.train_dataloader()

    # Load multiple batches
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Load 3 batches
            break
        batches.append(batch)

    # Check that we got the expected number of batches
    assert len(batches) == 3

    # Check that all batches have the correct shape
    for batch in batches:
        images, labels = batch
        assert images.shape == (4, 3, 64, 64)
        assert labels.shape[0] == 4
