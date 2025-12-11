import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from dataset import get_dataloaders, FEATURE_COLS
from models import RegimeMLP


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(batch_size=64)

    input_dim = test_loader.dataset.X.shape[1]
    num_classes = len(torch.unique(test_loader.dataset.y))

    model = RegimeMLP(input_dim=input_dim,
                      hidden_dims=[64, 32],
                      num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, preds = outputs.max(1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    print("Classification report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    # confusion matrix figure
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("checkpoints/confusion_matrix.png", bbox_inches="tight")

    # PCA plot
    df = pd.read_csv("data/processed/features_with_regime.csv")
    X = df[FEATURE_COLS].values
    y_regime = df["regime_id"].values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    for rid in np.unique(y_regime):
        idx = y_regime == rid
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Regime {rid}")
    plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Regimes")
    plt.savefig("checkpoints/pca_regimes.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
