# ================[ IMPORTS ]================
import torch

from src.models.components.time import TimeConditioner


# ================[ TESTS ]================
def test_time_conditioner_shapes():
    latent_dim = 64
    batch_size = 8
    num_latents = 16

    conditioner = TimeConditioner(latent_dim)
    z = torch.randn(batch_size, num_latents, latent_dim)
    t = torch.rand(batch_size)

    z_out = conditioner(z, t)

    assert z_out.shape == (batch_size, num_latents, latent_dim), "Output shape mismatch"


def test_time_conditioner_broadcast_and_effect():
    latent_dim = 32
    batch_size = 4
    num_latents = 10

    conditioner = TimeConditioner(latent_dim)
    z = torch.randn(batch_size, num_latents, latent_dim)
    t = torch.rand(batch_size)

    z_out = conditioner(z.clone(), t)

    # Should be different due to time conditioning
    assert not torch.allclose(z, z_out), "Time conditioning should alter z"

    # All latent positions within a sample should be shifted by the same time embedding
    delta = z_out - z
    diffs = delta - delta[:, 0:1, :]  # Compare all latents to the first latent's shift
    assert torch.allclose(diffs, torch.zeros_like(diffs), atol=1e-5), "Conditioning should broadcast across latents"
