# Stock Regime Classifier (EF2039 Project 2)

**Author:** Jungho Moon (20240110)  
**Course:** EF2039 AI Programming  

This project creates **market regimes** from cross-sectional return-based features using **KMeans (k=3)**, then trains a **PyTorch MLP** to classify each stock into a regime.

- Regimes: **Downtrend / Stable / High-Alpha**
- Features (6): `avg_return`, `volatility`, `skewness`, `kurtosis`, `mdd`, `sharpe`

---

## 1) Overview

A **regime** is a group of stocks that share similar **risk–return characteristics**.  
We first generate pseudo-labels via unsupervised clustering, then learn a supervised classifier for fast inference.

**Pipeline:** `StandardScaler → KMeans(k=3) → regime naming → MLP classifier`

---

## 2) Dataset

Input CSV (required):

- `data/features_filtered.csv`
- Required columns:
  - `Name`
  - `avg_return`, `volatility`, `skewness`, `kurtosis`, `mdd`, `sharpe`

Processed output:

- `data/processed/features_with_regime.csv`
- Added columns:
  - `regime_id`, `regime_name`

---

## 3) Regime Definitions (Interpretation)

We map KMeans cluster IDs to human-readable regimes by sorting clusters by mean `avg_return`.

- **Downtrend**
  - Negative average returns
  - High maximum drawdown (MDD)
- **Stable**
  - Moderate returns
  - Moderate risk
- **High-Alpha**
  - Higher returns
  - Relatively controlled risk profile

---

## 4) Model

**RegimeMLP architecture**

- Input: 6
- Hidden1: 6 → 64 (ReLU + Dropout 0.2)
- Hidden2: 64 → 32 (ReLU + Dropout 0.2)
- Output: 32 → 3 logits
- Loss: CrossEntropyLoss  
- Optimizer: Adam (`lr=1e-3`)

**Split**

- Train ~70% / Val ~15% / Test ~15% (stratified)

Artifacts saved:

- `checkpoints/scaler.pkl`
- `checkpoints/best_model.pt`

---

## 5) Quick Start

```bash
# 1) Create regimes (unsupervised labeling)
python src/create_regimes.py

# 2) Train classifier
python src/train.py --epochs 50 --batch_size 32 --lr 1e-3

# 3) Evaluate
python src/eval.py

# 4) Predict a single ticker
python src/predict.py --ticker "AAPL"
