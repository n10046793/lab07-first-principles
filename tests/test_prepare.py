"""Tests for the split, the preparation, the Dataset and the loaders."""

import torch

from prepare import SEED, make_loaders


def test_dataset_length_shapes_dtypes(splits):
    assert len(splits.train) == 3600
    assert len(splits.val) == 1200
    assert len(splits.test) == 1200
    x, y = splits.train[0]
    assert x.shape == (8,)
    assert y.shape == ()
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_split_sizes_no_overlap(splits):
    assert len(splits.train) + len(splits.val) + len(splits.test) == 6000
    train_rows = {tuple(row) for row in splits.train.x.tolist()}
    val_rows = {tuple(row) for row in splits.val.x.tolist()}
    test_rows = {tuple(row) for row in splits.test.x.tolist()}
    assert train_rows.isdisjoint(val_rows)
    assert train_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(test_rows)


def test_scaler_fitted_on_train_only(splits):
    # The first five columns are the scaled numbers.
    train_means = splits.train.x[:, :5].mean(dim=0)
    assert torch.allclose(train_means, torch.zeros(5), atol=1e-5)
    val_means = splits.val.x[:, :5].mean(dim=0)
    assert not torch.allclose(val_means, torch.zeros(5), atol=1e-5)


def test_loader_shuffling_repeatable(splits):
    first_loaders = make_loaders(splits, seed=SEED)
    second_loaders = make_loaders(splits, seed=SEED)
    first_x, _ = next(iter(first_loaders[0]))
    second_x, _ = next(iter(second_loaders[0]))
    assert torch.equal(first_x, second_x)
    # Shuffled, so the first batch is not the first rows in order.
    assert not torch.equal(first_x, splits.train.x[:64])
