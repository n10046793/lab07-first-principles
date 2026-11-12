"""Tests for the network."""

import pytest
import torch
from torch import nn

from model import BikeNet


@pytest.mark.parametrize("hidden", [4, 16, 32, 64])
def test_network_one_value_per_row(hidden):
    torch.manual_seed(42)
    model = BikeNet(n_features=8, hidden=hidden)
    x = torch.randn(5, 8)
    out = model(x)
    assert out.shape == (5,)


def test_one_optimizer_step_changes_every_layer(splits):
    torch.manual_seed(42)
    model = BikeNet(splits.n_features, hidden=32)
    before = {name: p.clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    x, y = splits.train[:64]
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for name, p in model.named_parameters():
        assert not torch.equal(p, before[name]), f"{name} did not change"
