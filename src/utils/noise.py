# ===============[ IMPORTS ]===============
from collections.abc import Callable

import torch


# ===============[ HELPERS ]===============
def gamma_linear(t: torch.Tensor) -> torch.Tensor:
    return 1.0 - t


def gamma_sigmoid(t: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(-5 + 10 * (1 - t))


def gamma_cosine(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t * torch.pi / 2) ** 2


def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample timesteps t ~ Uniform(0, 1) for a batch."""
    return torch.rand(batch_size, 1, 1, 1, device=device)  # broadcast to (B, 1, 1, 1)


# ===============[ CORE ]===============
def forward_noise(x0: torch.Tensor, t: torch.Tensor, gamma_fn: Callable) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply forward noising process to clean image x0 at timestep t.

    Parameters:
    - x0: clean image tensor of shape (B, C, H, W)
    - t: timestep tensor of shape (B,)
    - gamma_fn: function that computes gamma(t), returns shape (B,)

    Returns:
    - xt: noised image tensor of shape (B, C, H, W)
    - noise: the Gaussian noise that was added
    """
    B = x0.shape[0]
    gamma = gamma_fn(t).view(B, 1, 1, 1)
    noise = torch.randn_like(x0)
    xt = torch.sqrt(gamma) * x0 + torch.sqrt(1 - gamma) * noise
    return xt, noise


def denoise(
    xt: torch.Tensor, noise_pred: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, gamma_fn: Callable
) -> torch.Tensor:
    """Predict x_{t-1} using DDPM-style denoising and gamma schedule.

    Parameters:
    - xt: current noised image [B, C, H, W]
    - noise_pred: predicted noise [B, C, H, W]
    - t: current timestep [B]
    - t_next: next timestep [B]
    - gamma_fn: gamma schedule function

    Returns:
    - x_{t-1} estimate [B, C, H, W]
    """
    B = xt.shape[0]
    gamma_t = gamma_fn(t).view(B, 1, 1, 1)
    gamma_next = gamma_fn(t_next).view(B, 1, 1, 1)

    # Predict clean image x0
    x0_pred = (xt - torch.sqrt(1 - gamma_t) * noise_pred) / torch.sqrt(gamma_t)

    # Sample noise z for reverse process
    z = torch.randn_like(xt)

    # Re-noise x0 toward gamma(t_next)
    x_prev = torch.sqrt(gamma_next) * x0_pred + torch.sqrt(1 - gamma_next) * z
    return x_prev
