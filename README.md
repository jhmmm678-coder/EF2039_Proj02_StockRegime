Stock Regime Classifier
1. Overview

This project learns stock market regimes from cross-sectional return-based features and builds a neural network classifier that predicts the regime of each stock.

A regime is defined as a group of stocks that share similar risk–return characteristics. In this project, we consider three regimes:

Downtrend – negative average returns and high drawdowns

Stable – moderate risk and moderate returns

High-Alpha – higher returns with controlled risk

The goal is to provide a simple tool that assigns each stock to one of these regimes based on summary statistics of its daily returns.

2. Project Structure
EF2039_Proj02_StockRegime/
├─ data/
│  ├─ features_filtered.csv          # raw feature dataset (input)
│  └─ processed/
│       └─ features_with_regime.csv  # processed dataset with regime labels
├─ src/
│  ├─ create_regimes.py  # build regimes using KMeans clustering
│  ├─ dataset.py         # dataset loading and train/val/test split
│  ├─ models.py          # MLP model definition
│  ├─ train.py           # training script
│  ├─ eval.py            # evaluation and visualization
│  └─ predict.py         # inference for a single ticker
├─ checkpoints/          # saved scaler, best model, plots (ignored by git)
├─ README.md
└─ requirements.txt

3. Dataset Construction
3.1 Input Features

The project assumes an input CSV file:

data/features_filtered.csv


This file must contain at least the following columns:

Name – stock name or ticker symbol

avg_return – average daily return

volatility – standard deviation of daily returns

skewness – skewness of the return distribution

kurtosis – kurtosis of the return distribution

mdd – maximum drawdown

sharpe – Sharpe ratio (risk-adjusted return)

If the file is not encoded in UTF-8, it is read with a different encoding (e.g., latin1) to avoid decoding errors.

3.2 Regime Labeling (Unsupervised)

The script src/create_regimes.py constructs regimes in two steps:

Feature standardization

All numerical features
(avg_return, volatility, skewness, kurtosis, mdd, sharpe)
are standardized using StandardScaler.

KMeans clustering

KMeans with n_clusters = 3 and random_state = 42 is applied to the standardized features.
The resulting cluster IDs are stored in a new column regime_id.

To make the cluster labels interpretable, clusters are sorted by the average avg_return and mapped to human-readable regime names:

lowest average return → Downtrend

middle average return → Stable

highest average return → High-Alpha

The final processed dataset is saved as:

data/processed/features_with_regime.csv


with two new columns:

regime_id – integer cluster ID

regime_name – string label (Downtrend, Stable, High-Alpha)

You can build this processed dataset by running:

python src/create_regimes.py

4. Model Architecture

The classifier is a fully connected Multi-Layer Perceptron (MLP) implemented in PyTorch.

Input dimension: 6 features
(avg_return, volatility, skewness, kurtosis, mdd, sharpe)

Hidden layers:

Linear(6 → 64) + ReLU + Dropout(0.2)

Linear(64 → 32) + ReLU + Dropout(0.2)

Output layer:

Linear(32 → 3) → class logits for 3 regimes

Loss: Cross-Entropy Loss

Optimizer: Adam

Device: GPU (CUDA) if available; otherwise CPU

The model class is defined in src/models.py as RegimeMLP.

5. Training
5.1 Train/Validation/Test Split

The script src/dataset.py performs a stratified split of the processed dataset:

Train: ~70 %

Validation: ~15 %

Test: ~15 %

The split preserves the class distribution of regime_id.
Features are standardized using StandardScaler fitted only on the train set.
The fitted scaler is saved to:

checkpoints/scaler.pkl

5.2 Running Training
python src/train.py --epochs 50 --batch_size 32 --lr 1e-3


Arguments:

--epochs – number of training epochs (default: 50)

--batch_size – mini-batch size (default: 32)

--lr – learning rate (default: 1e-3)

The script prints the training and validation loss/accuracy at each epoch and saves the best model checkpoint (based on validation accuracy) to:

checkpoints/best_model.pt

6. Evaluation and Visualization

After training, you can evaluate the model on the test set and generate plots by running:

python src/eval.py


This script computes:

a classification report (precision, recall, F1-score per regime)

a confusion matrix on the test set

and generates the following plots in the checkpoints/ directory:

confusion_matrix.png – confusion matrix heatmap

pca_regimes.png – 2D PCA scatter plot of all stocks colored by regime_id

These visualizations are used to explain how well the classifier separates the regimes and how they are distributed in the feature space.

Example result summary (replace with your own numbers):

Test accuracy: XX.X %

Macro F1-score: YY.Y %

The High-Alpha regime tends to be (easier/harder) to classify because (your explanation here).

7. Inference Demo

You can predict the regime of a single stock using its name (ticker) in the processed CSV:

python src/predict.py --ticker "AAPL"


The script will:

Load data/processed/features_with_regime.csv.

Find the row whose Name matches the given ticker.

Standardize the feature vector using checkpoints/scaler.pkl.

Load the best model from checkpoints/best_model.pt.

Output:

predicted regime_id

predicted regime_name

class probabilities

a short textual description of the regime

Example output:

Ticker: AAPL
Predicted regime: High-Alpha (id=2)
Probabilities: [0.05 0.10 0.85]
Description: Higher return with controlled risk.

8. Installation and Requirements
8.1 Python Version

The project was developed and tested with:

Python 3.10 (other 3.x versions may also work)

8.2 Dependencies

All Python dependencies are listed in requirements.txt:

pandas
numpy
scikit-learn
torch
matplotlib
joblib


To install them, it is recommended to create a virtual environment:

python -m venv .venv
.\.venv\Scripts\activate   # on Windows
# source .venv/bin/activate   # on Linux / macOS
pip install -r requirements.txt

9. Limitations and Future Work

Limitations

Regimes are defined purely by unsupervised clustering; they are not validated against fundamental or macroeconomic labels.

Features are computed over a fixed look-back window, so regime definitions depend on the chosen period.

Only six summary statistics are used, which may not fully capture tail risk or intraday dynamics.

Future work

Add more features: beta, turnover, sector, valuation ratios, or factor exposures.

Use time-series models (e.g., temporal CNNs, LSTMs) to model regime transitions over time.

Backtest regime-based portfolios to evaluate real investment performance.

Extend to more regimes (e.g., “High-Volatility”, “Defensive”) or dynamic regime detection.

10. Reproducibility Checklist

To reproduce the full pipeline:

Prepare data/features_filtered.csv with the required columns.

Construct regimes and the processed dataset:

python src/create_regimes.py


Train the classifier:

python src/train.py


Evaluate and generate plots:

python src/eval.py


Run inference for a single ticker:

python src/predict.py --ticker "YOUR_TICKER_NAME"


This workflow covers the entire development cycle required in the assignment: idea, dataset construction, model design, training, performance analysis, code management, and distribution.