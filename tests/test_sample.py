# ================[ IMPORTS ]================
import pytest
import torch

from src.utils.noise import gamma_cosine
from src.utils.sample import sample_ddpm


# ================[ SETUP ]================
class DummyModel(torch.nn.Module):
    def forward(self, x, t):
        # Returns fake noise (zeros) with same shape
        return torch.zeros_like(x)


# ================[ TESTS ]================
@pytest.mark.parametrize("shape", [(2, 3, 32, 32), (4, 3, 64, 64)])
def test_sample_ddpm_output_shape(shape):
    model = DummyModel()
    output = sample_ddpm(model, shape=shape, gamma_fn=gamma_cosine, num_steps=10, device="cpu")
    assert output.shape == torch.Size(shape)


def test_sample_ddpm_no_nans():
    model = DummyModel()
    shape = (2, 3, 32, 32)
    output = sample_ddpm(model, shape=shape, gamma_fn=gamma_cosine, num_steps=10, device="cpu")
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
