# ================[ IMPORTS ]================
import pytest
import torch

from src.models.components.block import RINBlock


# ================[ SETUP ]================
@pytest.fixture
def rin_block():
    return RINBlock(latent_dim=32, interface_dim=64, num_heads=4, depth=2)


# ================[ TESTS ]================
def test_rin_block_output_shapes(rin_block):
    batch_size = 8
    num_latents = 16
    num_interface_tokens = 64

    Z = torch.randn(batch_size, num_latents, 32)
    X = torch.randn(batch_size, num_interface_tokens, 64)

    Z_out, X_out = rin_block(Z, X)

    assert Z_out.shape == Z.shape, "Latent token shape should be preserved"
    assert X_out.shape == X.shape, "Interface token shape should be preserved"


def test_rin_block_changes_values(rin_block):
    batch_size = 4
    num_latents = 10
    num_interface_tokens = 20

    Z = torch.randn(batch_size, num_latents, 32)
    X = torch.randn(batch_size, num_interface_tokens, 64)

    Z_out, X_out = rin_block(Z, X)

    assert not torch.allclose(Z, Z_out), "Latent tokens should change after RINBlock"
    assert not torch.allclose(X, X_out), "Interface tokens should change after RINBlock"


def test_rin_block_batch_independence(rin_block):
    batch_size = 2
    num_latents = 5
    num_interface_tokens = 15

    Z = torch.randn(batch_size, num_latents, 32)
    X = torch.randn(batch_size, num_interface_tokens, 64)

    Z_out, X_out = rin_block(Z, X)

    # Check batch independence
    assert not torch.allclose(Z_out[0], Z_out[1]), "Different batch elements should produce different latent outputs"
    assert not torch.allclose(X_out[0], X_out[1]), "Different batch elements should produce different interface outputs"
