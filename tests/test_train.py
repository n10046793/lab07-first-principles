"""Tests for the training loop."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import BikeNet
from prepare import BikeDataset
from train import run_epoch, train_model


def test_memorize_one_small_batch(splits):
    # If gradients were never cleared, or the optimizer step were
    # missing, the loss on one small batch would not fall like this.
    torch.manual_seed(42)
    small = BikeDataset(splits.train.x[:32].numpy(), splits.train.y[:32].numpy())
    loader = DataLoader(small, batch_size=32, shuffle=False)
    model = BikeNet(splits.n_features, hidden=32)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first_loss, _ = run_epoch(model, loader, loss_fn, optimizer)
    for _ in range(399):
        loss, _ = run_epoch(model, loader, loss_fn, optimizer)
    assert loss < 0.2 * first_loss


def test_validation_epoch_leaves_weights_unchanged(model, loaders):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    run_epoch(model, loaders[1], nn.MSELoss())
    after = model.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key])


def test_train_model_returns_history_and_restores_best(model, loaders):
    train_loader, val_loader, _ = loaders
    history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        nn.MSELoss(),
        torch.optim.Adam(model.parameters(), lr=3e-3),
        epochs=5,
    )
    assert len(history) == 5
    for row in history:
        assert set(row) == {"epoch", "train_loss", "val_loss", "val_mae"}
    assert 1 <= best_epoch <= 5
    best_val_loss = min(row["val_loss"] for row in history)
    val_loss, _ = run_epoch(model, val_loader, nn.MSELoss())
    assert abs(val_loss - best_val_loss) < 1e-4
