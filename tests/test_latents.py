# ================[ IMPORTS ]================
import torch
import torch.nn as nn

from src.models.components.latents import LatentInitializer


# ================[ TESTS ]================
def test_forward_latents_shape_and_consistency():
    latent_dim = 64
    num_latents = 16
    batch_size = 8

    init = LatentInitializer(num_latents=num_latents, latent_dim=latent_dim)
    z1 = init.forward(batch_size)
    _ = init.forward(batch_size)

    # Check shape
    assert z1.shape == (batch_size, num_latents, latent_dim)

    # Check that all batch entries are identical (because expand not repeat)
    for i in range(1, batch_size):
        assert torch.allclose(z1[0], z1[i]), "All rows should be equal copies of z_init"


def test_warm_start_returns_init_when_norm_is_zero():
    """When LayerNorm is zero-initialized, warm_start should return exactly z_init"""
    latent_dim = 64
    num_latents = 16
    batch_size = 4

    init = LatentInitializer(num_latents=num_latents, latent_dim=latent_dim)
    z_prev = torch.randn(batch_size, num_latents, latent_dim)

    z_warm = init.warm_start(z_prev)
    z_init = init.forward(batch_size)

    # Should be identical (or very close)
    assert torch.allclose(z_warm, z_init, atol=1e-6), "Warm start with zero-initialized LayerNorm should equal z_init"


def test_warm_start_changes_output_after_layernorm_updates():
    """If LayerNorm has learned weights, warm_start should diverge from z_init"""
    latent_dim = 64
    num_latents = 16
    batch_size = 4

    init = LatentInitializer(num_latents=num_latents, latent_dim=latent_dim)

    # Manually change LayerNorm to simulate learning
    nn.init.ones_(init.layer_norm.weight)
    nn.init.zeros_(init.layer_norm.bias)

    z_prev = torch.randn(batch_size, num_latents, latent_dim)
    z_warm = init.warm_start(z_prev)
    z_init = init.forward(batch_size)

    assert not torch.allclose(z_warm, z_init), "With non-zero LayerNorm weights, warm start should differ from z_init"


def test_warm_start_early_training_identity_behavior():
    """With LayerNorm weights and bias zeroed, warm start should initially return something close to z_init."""
    latent_dim = 32
    num_latents = 8
    batch_size = 2

    init = LatentInitializer(num_latents=num_latents, latent_dim=latent_dim)

    z_prev = torch.randn(batch_size, num_latents, latent_dim)
    z_init = init.forward(batch_size)
    z_warm = init.warm_start(z_prev)

    delta = z_warm - z_init
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-4), "Early in training, warm start ≈ z_init"
