import torch

from baselines.SambaY.models.SambaY import SambaYGMU


def test_sambay_gmu_shape_and_gradients():
    gmu = SambaYGMU(d_model=8, d_mem=16)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    memory = torch.randn(2, 4, 16)

    output = gmu(hidden, memory)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == hidden.shape
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()

