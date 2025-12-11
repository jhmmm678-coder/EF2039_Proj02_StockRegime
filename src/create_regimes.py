import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RANDOM_SEED = 42

def main():
    data_path = os.path.join("data", "features_filtered.xlsx")  # ← 엑셀로
    df = pd.read_excel(data_path)  # read_csv 말고 read_excel

    feature_cols = ['avg_return', 'volatility', 'skewness',
                    'kurtosis', 'mdd', 'sharpe']
    
    # 혹시 모를 공백/BOM 제거
    df.columns = (
        df.columns.astype(str)
                  .str.strip()
                  .str.replace('\ufeff', '', regex=False)
    )

    # 안전하게: 빠진 컬럼 체크
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"다음 컬럼이 없습니다: {missing}\n현재 컬럼: {df.columns.tolist()}")

    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)
    df['regime_id'] = cluster_ids

    cluster_stats = df.groupby('regime_id')['avg_return'].mean().sort_values()
    ordered_clusters = cluster_stats.index.tolist()

    regime_name_map = {}
    for i, cid in enumerate(ordered_clusters):
        if i == 0:
            regime_name_map[cid] = "Downtrend"
        elif i == 1:
            regime_name_map[cid] = "Stable"
        else:
            regime_name_map[cid] = "High-Alpha"

    df['regime_name'] = df['regime_id'].map(regime_name_map)

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    out_path = os.path.join("data", "processed", "features_with_regime.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved processed dataset to {out_path}")

if __name__ == "__main__":
    main()
