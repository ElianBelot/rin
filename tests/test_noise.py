# ================[ IMPORTS ]================
import pytest
import torch

from src.utils.noise import denoise, forward_noise, gamma_cosine, gamma_linear, gamma_sigmoid, sample_timesteps


# ================[ TESTS ]================
def test_gamma_linear():
    t = torch.tensor([0.0, 0.5, 1.0])
    gamma = gamma_linear(t)
    expected = torch.tensor([1.0, 0.5, 0.0])
    assert torch.allclose(gamma, expected)


def test_gamma_sigmoid_shape_and_range():
    t = torch.linspace(0, 1, 10)
    gamma = gamma_sigmoid(t)
    assert gamma.shape == t.shape
    assert torch.all((gamma >= 0) & (gamma <= 1))


def test_sample_timesteps():
    batch_size = 4
    t = sample_timesteps(batch_size, device="cpu")
    assert t.shape == (batch_size, 1, 1, 1)
    assert torch.all((t >= 0) & (t <= 1))


def test_noise_shapes():
    B, C, H, W = 4, 3, 64, 64
    x0 = torch.randn(B, C, H, W)
    t = torch.rand(B, 1, 1, 1)
    xt, eps = forward_noise(x0, t, gamma_fn=gamma_linear)
    assert xt.shape == (B, C, H, W)
    assert eps.shape == (B, C, H, W)


def test_noise_output_range():
    B, C, H, W = 4, 3, 64, 64
    x0 = torch.ones(B, C, H, W)
    t = torch.full((B, 1, 1, 1), 0.5)
    xt, eps = forward_noise(x0, t, gamma_fn=gamma_linear)
    assert torch.isfinite(xt).all()
    assert torch.isfinite(eps).all()


@pytest.mark.parametrize("gamma_fn", [gamma_linear, gamma_sigmoid, gamma_cosine])
def test_denoise_shapes(gamma_fn):
    B, C, H, W = 4, 3, 64, 64
    xt = torch.randn(B, C, H, W)
    noise_pred = torch.randn(B, C, H, W)
    t = torch.rand(B)
    t_next = torch.clamp(t - 0.01, min=0.0)  # ensure t_next < t

    x_prev = denoise(xt, noise_pred, t, t_next, gamma_fn)

    assert isinstance(x_prev, torch.Tensor)
    assert x_prev.shape == (B, C, H, W)


def test_denoise_output_numerics():
    B, C, H, W = 4, 3, 64, 64
    xt = torch.ones(B, C, H, W)
    noise_pred = torch.zeros(B, C, H, W)
    t = torch.tensor([0.5] * B)
    t_next = torch.tensor([0.4] * B)

    x_prev = denoise(xt, noise_pred, t, t_next, gamma_cosine)
    assert torch.isfinite(x_prev).all()


def test_denoise_edge_case_near_t0():
    B, C, H, W = 4, 3, 64, 64
    xt = torch.randn(B, C, H, W)
    noise_pred = torch.randn(B, C, H, W)
    t = torch.full((B,), 0.01)
    t_next = torch.zeros_like(t)

    x_prev = denoise(xt, noise_pred, t, t_next, gamma_cosine)
    assert x_prev.shape == (B, C, H, W)
