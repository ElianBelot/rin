# ===============[ IMPORTS ]===============
from collections.abc import Callable

import torch
import torch.nn as nn

from src.utils.noise import denoise


# ===============[ CORE ]===============
@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    shape: tuple[int, int, int, int],
    gamma_fn: Callable,
    num_steps: int = 50,
    device: torch.device = "cuda",
) -> torch.Tensor:
    """Generate samples using DDPM sampling with latent self-conditioning (RIN).

    Parameters:
    - model: the DDPM model predicting noise and latent tokens, `net(x, t, Z_prev)`
    - shape: output shape, e.g. (B, 3, 64, 64)
    - gamma_fn: gamma schedule function
    - num_steps: number of diffusion steps
    - device: torch device

    Returns:
    - Final image tensor x0 [B, C, H, W]
    """
    model.eval()
    B = shape[0]
    # FIXME: Horrible
    if device.type == "cuda":
        x = torch.randn(*shape, device=device, dtype=torch.float32)
    else:
        x = torch.randn(*shape, device=device)
    Z_prev = None

    # Create timesteps decreasing from 1.0 -> 0.0
    # FIXME: Horrible
    if device.type == "cuda":
        t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    else:
        t_vals = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    # Iterate over timesteps
    for i in range(num_steps):
        t = t_vals[i].expand(B)
        t_next = t_vals[i + 1].expand(B)

        # Predict noise and update latent tokens (self-conditioning)
        noise_pred, Z_prev, _ = model(x, t, Z_prev)

        # Denoise step
        x = denoise(x, noise_pred, t, t_next, gamma_fn)

    return x
