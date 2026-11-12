"""Shared fixtures for the Lab 07 tests."""

import pytest
import torch

from model import BikeNet
from prepare import SEED, make_loaders, make_splits


@pytest.fixture(scope="module")
def splits():
    return make_splits(SEED)


@pytest.fixture(scope="module")
def loaders(splits):
    return make_loaders(splits)


@pytest.fixture
def model(splits):
    torch.manual_seed(SEED)
    return BikeNet(splits.n_features, hidden=32)
