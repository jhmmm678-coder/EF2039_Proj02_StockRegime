import os
import argparse
import pandas as pd
import numpy as np
import torch
import joblib

from models import RegimeMLP
from dataset import FEATURE_COLS

REGIME_DESC = {
    "Downtrend": "High drawdown and negative return (risky regime).",
    "Stable": "Moderate risk and moderate return.",
    "High-Alpha": "Higher return with controlled risk."
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv("data/processed/features_with_regime.csv")
    row = df[df["Name"] == args.ticker].iloc[0]

    scaler = joblib.load(os.path.join("checkpoints", "scaler.pkl"))

    x = row[FEATURE_COLS].values.reshape(1, -1)
    x_scaled = scaler.transform(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(FEATURE_COLS)
    num_classes = df["regime_id"].nunique()

    model = RegimeMLP(input_dim=input_dim,
                      hidden_dims=[64, 32],
                      num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval()

    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    # regime_id -> regime_name
    name_map = df.drop_duplicates("regime_id").set_index("regime_id")["regime_name"].to_dict()
    regime_name = name_map[pred_class]
    desc = REGIME_DESC.get(regime_name, "")

    print(f"Ticker: {row['Name']}")
    print(f"Predicted regime: {regime_name} (id={pred_class})")
    print(f"Probabilities: {probs}")
    print(f"Description: {desc}")


if __name__ == "__main__":
    main()
