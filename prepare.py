"""Split, prepare, Dataset and loaders for Lakeshore Bike Share.

The split is made first, before any preparation. The scaler and the
one-hot encoder are fitted on the training rows only, so the validation
and test rows do not leak into training.
"""

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]
FEATURES = NUMERIC + CATEGORICAL
SEED = 42
BATCH_SIZE = 64


class Splits:
    """The three splits, with prepared arrays and the fitted preparation."""

    def __init__(self, train, val, test, preprocessor):
        self.train = train
        self.val = val
        self.test = test
        self.preprocessor = preprocessor
        self.n_features = train.x.shape[1]


class BikeDataset(Dataset):
    """One row at a time, as float32 tensors."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def make_preprocessor() -> ColumnTransformer:
    """Scale the numbers and one-hot encode the station type."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )


def make_splits(seed: int = SEED) -> Splits:
    """Split 60/20/20, then fit the preparation on the training rows only."""
    frame = make_rentals()
    train_df, rest_df = train_test_split(frame, test_size=0.4, random_state=seed)
    val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=seed)

    preprocessor = make_preprocessor()
    preprocessor.fit(train_df[FEATURES])

    def make_dataset(df):
        features = preprocessor.transform(df[FEATURES])
        labels = df[TARGET].to_numpy(dtype=float)
        return BikeDataset(features, labels)

    return Splits(
        make_dataset(train_df),
        make_dataset(val_df),
        make_dataset(test_df),
        preprocessor,
    )


def make_loaders(splits: Splits, batch_size: int = BATCH_SIZE, seed: int = SEED):
    """Training batches are shuffled, validation and test batches are not."""
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(splits.val, batch_size=256, shuffle=False)
    test_loader = DataLoader(splits.test, batch_size=256, shuffle=False)
    return train_loader, val_loader, test_loader
