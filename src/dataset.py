import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import torch
from torch.utils.data import Dataset, DataLoader

FEATURE_COLS = ['avg_return', 'volatility', 'skewness',
                'kurtosis', 'mdd', 'sharpe']
LABEL_COL = 'regime_id'
RANDOM_SEED = 42


class RegimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data_and_split(test_size=0.15, val_size=0.15):
    data_path = os.path.join("data", "processed", "features_with_regime.csv")
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size),
        random_state=RANDOM_SEED, stratify=y
    )

    relative_val_size = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_val_size,
        random_state=RANDOM_SEED, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("checkpoints", exist_ok=True)
    joblib.dump(scaler, os.path.join("checkpoints", "scaler.pkl"))

    return (X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test)


def get_dataloaders(batch_size=32):
    (X_train, y_train,
     X_val, y_val,
     X_test, y_test) = load_data_and_split()

    train_ds = RegimeDataset(X_train, y_train)
    val_ds = RegimeDataset(X_val, y_val)
    test_ds = RegimeDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
