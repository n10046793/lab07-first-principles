"""Train the Lakeshore Bike Share network and write every result.

Run with: python train.py

This makes the reports folder again from scratch: the learning curves,
the per-epoch history, the experiments table and the metrics file.
The seed is fixed, so running it twice on the same computer gives the
same numbers.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn

from model import BikeNet
from prepare import BATCH_SIZE, SEED, make_loaders, make_splits

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
EPOCHS = 80
LEARNING_RATE = 3e-3
HIDDEN = 32


def run_epoch(model, loader, loss_fn, optimizer=None):
    """Run one pass over a loader.

    With an optimizer, this trains. Without one, it only measures.
    Returns the mean loss and the mean absolute error, in rentals.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_abs = 0.0
    count = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            total_abs += torch.abs(logits - y).sum().item()
            count += len(y)
    return total_loss / count, total_abs / count


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    """Train for many epochs and put back the best weights.

    The best epoch is the one with the lowest validation loss.
    Returns the history, one row per epoch, and the best epoch number.
    """
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:3d} train loss {train_loss:8.2f} "
            f"val loss {val_loss:8.2f} val MAE {val_mae:6.2f}"
        )
    model.load_state_dict(best_state)
    return history, best_epoch


def plot_curves(history, path, best_epoch):
    """Plot train and val loss per epoch and mark the best epoch."""
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    plt.figure()
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.axvline(best_epoch, color="grey", linestyle="--", label="best epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss (mean squared error)")
    plt.title("Learning curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def run_one(name, hidden, learning_rate, epochs, splits):
    """Train one network and return its history and best epoch."""
    torch.manual_seed(SEED)
    train_loader, val_loader, _ = make_loaders(splits, BATCH_SIZE, SEED)
    model = BikeNet(splits.n_features, hidden=hidden)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(f"\nRun: {name}")
    history, best_epoch = train_model(
        model, train_loader, val_loader, loss_fn, optimizer, epochs
    )
    best_row = min(history, key=lambda row: row["val_loss"])
    return (
        {
            "name": name,
            "hidden": hidden,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_mae": best_row["val_mae"],
            "best_val_loss": best_row["val_loss"],
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
        },
        history,
        model,
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    REPORTS.mkdir(exist_ok=True)

    splits = make_splits(SEED)
    print(f"train {len(splits.train)} val {len(splits.val)} test {len(splits.test)}")
    print(f"features per row: {splits.n_features}")

    # Main run.
    main_result, history, model = run_one("main", HIDDEN, LEARNING_RATE, EPOCHS, splits)
    plot_curves(history, REPORTS / "learning_curves.png", main_result["best_epoch"])
    with open(REPORTS / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae"]
        )
        writer.writeheader()
        writer.writerows(history)

    # Baselines, on the same rows, after the best weights are restored.
    _, val_loader, test_loader = make_loaders(splits, BATCH_SIZE, SEED)
    loss_fn = nn.MSELoss()
    val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
    test_loss, test_mae = run_epoch(model, test_loader, loss_fn)

    train_mean = float(splits.train.y.mean().item())
    val_mean_mae = float(
        mean_absolute_error(splits.val.y.numpy(), np.full(len(splits.val), train_mean))
    )
    test_mean_mae = float(
        mean_absolute_error(
            splits.test.y.numpy(), np.full(len(splits.test), train_mean)
        )
    )

    linear = LinearRegression()
    linear.fit(splits.train.x.numpy(), splits.train.y.numpy())
    val_linear_mae = float(
        mean_absolute_error(splits.val.y.numpy(), linear.predict(splits.val.x.numpy()))
    )
    test_linear_mae = float(
        mean_absolute_error(
            splits.test.y.numpy(), linear.predict(splits.test.x.numpy())
        )
    )

    metrics = {
        "n_features": splits.n_features,
        "train_rows": len(splits.train),
        "val_rows": len(splits.val),
        "test_rows": len(splits.test),
        "main": main_result,
        "network_val_mae": val_mae,
        "network_test_mae": test_mae,
        "network_val_loss": val_loss,
        "network_test_loss": test_loss,
        "mean_baseline_val_mae": val_mean_mae,
        "mean_baseline_test_mae": test_mean_mae,
        "linear_baseline_val_mae": val_linear_mae,
        "linear_baseline_test_mae": test_linear_mae,
    }
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nBest epoch: {main_result['best_epoch']}")
    print(f"Network val MAE: {val_mae:.2f} test MAE: {test_mae:.2f}")
    print(f"Mean baseline val MAE: {val_mean_mae:.2f} test MAE: {test_mean_mae:.2f}")
    print(
        f"Linear baseline val MAE: {val_linear_mae:.2f} test MAE: {test_linear_mae:.2f}"
    )

    # Experiments: each one changes one thing from the main run.
    experiments = [main_result]
    for name, hidden, learning_rate in [
        ("learning rate too small", HIDDEN, 1e-4),
        ("learning rate too large", HIDDEN, 3e-2),
        ("wider network", 128, LEARNING_RATE),
    ]:
        result, exp_history, _ = run_one(name, hidden, learning_rate, EPOCHS, splits)
        experiments.append(result)
        safe_name = name.replace(" ", "_")
        plot_curves(
            exp_history,
            REPORTS / f"learning_curves_{safe_name}.png",
            result["best_epoch"],
        )
    with open(REPORTS / "experiments.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "hidden",
                "learning_rate",
                "epochs",
                "best_epoch",
                "best_val_mae",
                "best_val_loss",
                "final_train_loss",
                "final_val_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(experiments)
    print("\nDone. Reports are in reports/.")


if __name__ == "__main__":
    main()
