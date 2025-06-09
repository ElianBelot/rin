# ===============[ IMPORTS ]===============
import pytest
import torch

from src.models.rin import RINModel


# ===============[ SETUP ]===============
@pytest.fixture
def rin_model():
    return RINModel(
        image_size=64,
        patch_size=8,
        latent_dim=32,
        interface_dim=64,
        num_latents=16,
        num_blocks=2,
        block_depth=1,
        num_heads=4,
    )


# ===============[ TESTS ]===============
def test_rin_model_output_shape(rin_model):
    batch_size = 4
    image_size = 64

    x = torch.randn(batch_size, 3, image_size, image_size)
    t = torch.rand(batch_size)

    output = rin_model(x, t)

    assert output.shape == (batch_size, 3, image_size, image_size), "Output shape mismatch."


def test_rin_model_with_latent_self_conditioning(rin_model):
    batch_size = 2
    image_size = 64
    latent_dim = rin_model.latent_dim
    num_latents = rin_model.num_latents

    x = torch.randn(batch_size, 3, image_size, image_size)
    t = torch.rand(batch_size)
    Z_prev = torch.randn(batch_size, num_latents, latent_dim)

    output = rin_model(x, t, Z_prev=Z_prev)

    assert output.shape == (batch_size, 3, image_size, image_size), "Output shape mismatch with latent conditioning."


def test_rin_model_changes_input(rin_model):
    batch_size = 1
    image_size = 64

    x = torch.randn(batch_size, 3, image_size, image_size)
    t = torch.rand(batch_size)

    output = rin_model(x, t)

    assert not torch.allclose(x, output), "Model output should differ from input."


def test_rin_model_batch_independence(rin_model):
    batch_size = 2
    image_size = 64

    x = torch.randn(batch_size, 3, image_size, image_size)
    t = torch.rand(batch_size)

    output = rin_model(x, t)

    assert not torch.allclose(output[0], output[1]), "Outputs for different batch items should differ."
