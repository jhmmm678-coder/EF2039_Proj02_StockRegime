<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="color-scheme" content="dark light" />
  <title>Stock Regime Classifier | EF2039 Project 2</title>

  <style>
    :root{
      --bg: #0b1020;
      --panel: rgba(255,255,255,.06);
      --panel2: rgba(255,255,255,.04);
      --line: rgba(255,255,255,.10);
      --text: rgba(255,255,255,.92);
      --muted: rgba(255,255,255,.72);
      --muted2: rgba(255,255,255,.58);
      --shadow: 0 14px 40px rgba(0,0,0,.35);
      --radius: 18px;

      --a: #7aa7ff;    /* accent */
      --b: #5eead4;    /* accent 2 */
      --c: #a78bfa;    /* accent 3 */
      --warn: #ffcf5b;
      --bad: #ff6b6b;
      --good: #6ee7b7;

      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
    }

    /* Light mode (optional) */
    [data-theme="light"]{
      --bg: #f7f8fc;
      --panel: rgba(20,30,60,.06);
      --panel2: rgba(20,30,60,.045);
      --line: rgba(20,30,60,.12);
      --text: rgba(10,16,30,.92);
      --muted: rgba(10,16,30,.70);
      --muted2: rgba(10,16,30,.56);
      --shadow: 0 16px 46px rgba(10,16,30,.14);
    }

    *{ box-sizing:border-box; }
    html{ scroll-behavior:smooth; }
    body{
      margin:0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(1100px 520px at 15% -10%, color-mix(in srgb, var(--a) 26%, transparent), transparent 60%),
        radial-gradient(900px 520px at 100% 10%, color-mix(in srgb, var(--b) 18%, transparent), transparent 55%),
        radial-gradient(900px 560px at 50% 115%, color-mix(in srgb, var(--c) 14%, transparent), transparent 60%),
        var(--bg);
    }

    a{ color: var(--a); text-decoration:none; }
    a:hover{ text-decoration:underline; }

    .wrap{
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px 16px 54px;
    }

    /* Top bar */
    .top{
      position: sticky;
      top: 10px;
      z-index: 50;
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: color-mix(in srgb, var(--panel) 85%, transparent);
      backdrop-filter: blur(10px);
      box-shadow: 0 10px 26px rgba(0,0,0,.18);
    }

    .brand{
      display:flex;
      align-items:center;
      gap:10px;
      min-width: 220px;
    }
    .logo{
      width: 34px; height: 34px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--a), var(--b));
      box-shadow: 0 10px 26px rgba(0,0,0,.20);
    }
    .brand .t1{ font-weight: 800; letter-spacing:.2px; line-height:1.05; }
    .brand .t2{ font-size:12px; color: var(--muted2); margin-top:1px; }

    .nav{
      display:flex;
      gap: 12px;
      flex-wrap: wrap;
      justify-content:center;
      align-items:center;
      font-size: 13px;
      color: var(--muted);
    }
    .nav a{ color: var(--muted); padding:6px 8px; border-radius: 999px; }
    .nav a:hover{ background: var(--panel2); text-decoration:none; color: var(--text); }

    .actions{
      display:flex; align-items:center; gap:8px;
      min-width: 240px;
      justify-content:flex-end;
    }
    .btn{
      display:inline-flex; align-items:center; gap:8px;
      padding: 8px 11px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--panel2);
      color: var(--text);
      font-size: 13px;
      cursor:pointer;
      user-select:none;
      transition: transform .12s ease, background .12s ease;
    }
    .btn:hover{ background: var(--panel); transform: translateY(-1px); text-decoration:none; }
    .btn.primary{
      border-color: color-mix(in srgb, var(--a) 60%, var(--line));
      background: linear-gradient(135deg, color-mix(in srgb, var(--a) 32%, transparent), color-mix(in srgb, var(--b) 22%, transparent));
    }
    .btn small{ color: var(--muted2); }

    /* Hero */
    .hero{
      margin-top: 16px;
      padding: 20px;
      border-radius: var(--radius);
      border: 1px solid var(--line);
      background: color-mix(in srgb, var(--panel) 90%, transparent);
      box-shadow: var(--shadow);
      overflow:hidden;
      position:relative;
    }
    .hero::before{
      content:"";
      position:absolute; inset:-1px;
      background:
        radial-gradient(700px 220px at 20% 0%, color-mix(in srgb, var(--a) 22%, transparent), transparent 65%),
        radial-gradient(520px 220px at 88% 15%, color-mix(in srgb, var(--b) 18%, transparent), transparent 60%);
      pointer-events:none;
      opacity:.9;
    }
    .hero > *{ position:relative; z-index:1; }

    .heroGrid{
      display:grid;
      grid-template-columns: 1.35fr .65fr;
      gap: 14px;
      align-items: start;
    }
    @media (max-width: 980px){
      .heroGrid{ grid-template-columns: 1fr; }
      .actions{ min-width:auto; }
      .brand{ min-width:auto; }
    }

    h1{
      margin: 0 0 10px;
      font-size: 36px;
      line-height: 1.1;
      letter-spacing: .2px;
    }
    .subtitle{
      margin: 0;
      color: var(--muted);
      font-size: 15.5px;
      line-height: 1.6;
      max-width: 74ch;
    }

    .chips{
      display:flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 14px;
    }
    .chip{
      display:inline-flex;
      gap:8px;
      align-items:center;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--panel2);
      color: var(--muted);
      font-size: 13px;
    }
    .dot{ width:9px; height:9px; border-radius:999px; display:inline-block; background: var(--b); }

    .metaCard{
      border:1px solid var(--line);
      border-radius: var(--radius);
      background: color-mix(in srgb, var(--panel2) 95%, transparent);
      padding: 14px;
    }
    .metaRow{ display:flex; justify-content:space-between; gap:10px; padding:8px 0; border-bottom:1px dashed var(--line); }
    .metaRow:last-child{ border-bottom:none; }
    .metaRow span{ color: var(--muted2); font-size: 12px; }
    .metaRow b{ font-size: 13.5px; }

    /* Sections & cards */
    .section{
      margin-top: 16px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel2);
      padding: 16px;
    }
    .section h2{
      margin:0 0 10px;
      font-size: 18px;
      letter-spacing:.2px;
    }
    .section p{ margin: 8px 0 0; color: var(--muted); }

    .grid2{
      margin-top: 14px;
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    @media (max-width: 980px){ .grid2{ grid-template-columns: 1fr; } }

    .card{
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: color-mix(in srgb, var(--panel) 86%, transparent);
      padding: 14px;
      box-shadow: 0 12px 30px rgba(0,0,0,.12);
    }
    .card h3{ margin: 0 0 8px; font-size: 15px; }
    .card ul{ margin: 8px 0 0 18px; color: var(--muted); }
    .card li{ margin: 6px 0; }

    .kpis{
      display:grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-top: 12px;
    }
    @media (max-width: 980px){ .kpis{ grid-template-columns: 1fr 1fr; } }
    @media (max-width: 520px){ .kpis{ grid-template-columns: 1fr; } }

    .kpi{
      padding: 12px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg,
        color-mix(in srgb, var(--a) 12%, transparent),
        color-mix(in srgb, var(--b) 10%, transparent));
    }
    .kpi .label{ font-size: 12px; color: var(--muted2); }
    .kpi .value{ margin-top: 6px; font-size: 18px; font-weight: 800; }
    .kpi .hint{ margin-top: 6px; font-size: 12px; color: var(--muted2); }

    .code{
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(0,0,0,.22);
      padding: 12px;
      font-family: var(--mono);
      font-size: 12.8px;
      color: color-mix(in srgb, var(--text) 92%, transparent);
      overflow:auto;
    }

    .pipeline{
      display:grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-top: 12px;
    }
    @media (max-width: 980px){ .pipeline{ grid-template-columns: 1fr 1fr; } }
    @media (max-width: 520px){ .pipeline{ grid-template-columns: 1fr; } }

    .step{
      border:1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      background: color-mix(in srgb, var(--panel) 86%, transparent);
      position:relative;
      overflow:hidden;
    }
    .step::before{
      content:"";
      position:absolute; inset:-1px;
      background: radial-gradient(500px 150px at 10% 0%, color-mix(in srgb, var(--c) 16%, transparent), transparent 60%);
      opacity:.55;
      pointer-events:none;
    }
    .step > *{ position:relative; z-index:1; }
    .step .num{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      width: 28px; height: 28px;
      border-radius: 10px;
      background: rgba(255,255,255,.08);
      border:1px solid var(--line);
      font-weight: 800;
      margin-bottom: 8px;
    }
    .step b{ display:block; margin-bottom:6px; }
    .step p{ margin:0; color: var(--muted); font-size: 13px; line-height:1.5; }

    .figGrid{
      margin-top: 12px;
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    @media (max-width: 980px){ .figGrid{ grid-template-columns: 1fr; } }

    .figure{
      border:1px solid var(--line);
      border-radius: var(--radius);
      background: color-mix(in srgb, var(--panel) 88%, transparent);
      padding: 10px;
    }
    .figure .ph{
      border-radius: 14px;
      border:1px solid color-mix(in srgb, var(--line) 90%, transparent);
      background:
        linear-gradient(135deg,
          color-mix(in srgb, var(--a) 18%, transparent),
          color-mix(in srgb, var(--b) 12%, transparent));
      height: 260px;
      display:flex;
      align-items:center;
      justify-content:center;
      color: var(--muted2);
      font-size: 13px;
      text-align:center;
      padding: 14px;
    }
    .figure img{
      width: 100%;
      display:block;
      border-radius: 14px;
      border:1px solid color-mix(in srgb, var(--line) 90%, transparent);
    }
    .cap{
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted2);
      line-height:1.45;
    }

    .footer{
      margin-top: 16px;
      padding: 14px;
      border-radius: var(--radius);
      border: 1px solid var(--line);
      background: var(--panel2);
      color: var(--muted2);
      font-size: 12.5px;
      display:flex;
      justify-content:space-between;
      gap: 10px;
      flex-wrap:wrap;
    }
  </style>
</head>

<body>
  <div class="wrap">
    <div class="top">
      <div class="brand">
        <div class="logo" aria-hidden="true"></div>
        <div>
          <div class="t1">Stock Regime Classifier</div>
          <div class="t2">EF2039 AI Programming · Project 2</div>
        </div>
      </div>

      <div class="nav">
        <a href="#overview">Overview</a>
        <a href="#data">Data</a>
        <a href="#pipeline">Pipeline</a>
        <a href="#model">Model</a>
        <a href="#results">Results</a>
        <a href="#demo">Demo</a>
        <a href="#limits">Limitations</a>
      </div>

      <div class="actions">
        <a class="btn primary" href="https://github.com/jhmmm678-coder/EF2039_Proj02_StockRegime" target="_blank" rel="noreferrer">
          Repo
        </a>
        <button class="btn" id="themeBtn" type="button" aria-label="Toggle theme">
          Theme <small id="themeLabel">(auto)</small>
        </button>
      </div>
    </div>

    <section class="hero" id="overview">
      <div class="heroGrid">
        <div>
          <h1>Learn market regimes, then classify stocks.</h1>
          <p class="subtitle">
            This project builds regimes from cross-sectional <b>return-based features</b> and trains a
            <b>PyTorch MLP</b> that predicts the regime of each stock. A regime represents a group of stocks
            with similar <b>risk–return characteristics</b>.
          </p>

          <div class="chips" aria-label="Key highlights">
            <span class="chip"><span class="dot"></span>Unsupervised labeling (KMeans, k=3)</span>
            <span class="chip"><span class="dot"></span>Supervised classifier (MLP)</span>
            <span class="chip"><span class="dot"></span>6 summary statistics</span>
            <span class="chip"><span class="dot"></span>3 regimes: Downtrend / Stable / High-Alpha</span>
          </div>

          <div class="section" style="margin-top:14px;">
            <h2>Regime Definitions</h2>
            <div class="grid2">
              <div class="card">
                <h3>Downtrend</h3>
                <ul>
                  <li>Negative average returns</li>
                  <li>High maximum drawdowns (MDD)</li>
                </ul>
              </div>
              <div class="card">
                <h3>Stable</h3>
                <ul>
                  <li>Moderate risk</li>
                  <li>Moderate returns</li>
                </ul>
              </div>
              <div class="card">
                <h3>High-Alpha</h3>
                <ul>
                  <li>Higher returns</li>
                  <li>Controlled risk profile</li>
                </ul>
              </div>
              <div class="card">
                <h3>Goal</h3>
                <ul>
                  <li>Assign each stock to one regime</li>
                  <li>Enable quick, interpretable screening</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <aside class="metaCard" aria-label="Project metadata">
          <div class="metaRow"><span>Author</span><b>Jungho Moon (20240110)</b></div>
          <div class="metaRow"><span>Course</span><b>EF2039 AI Programming</b></div>
          <div class="metaRow"><span>Stack</span><b>pandas · sklearn · torch · matplotlib</b></div>
          <div class="metaRow"><span>Labeling</span><b>StandardScaler → KMeans (k=3)</b></div>
          <div class="metaRow"><span>Classifier</span><b>MLP (6→64→32→3)</b></div>
          <div class="metaRow"><span>Artifacts</span><b>Confusion matrix · PCA plot</b></div>

          <div class="section" style="margin-top:12px;">
            <h2 style="margin-bottom:6px;">Results Snapshot</h2>
            <div class="kpis">
              <div class="kpi">
                <div class="label">Test Accuracy</div>
                <div class="value">XX.X%</div>
                <div class="hint">Replace with your eval result</div>
              </div>
              <div class="kpi">
                <div class="label">Macro F1</div>
                <div class="value">YY.Y%</div>
                <div class="hint">Replace with your eval result</div>
              </div>
              <div class="kpi">
                <div class="label">#Features</div>
                <div class="value">6</div>
                <div class="hint">avg_return … sharpe</div>
              </div>
              <div class="kpi">
                <div class="label">#Regimes</div>
                <div class="value">3</div>
                <div class="hint">Down/Stable/Alpha</div>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </section>

    <section class="section" id="data">
      <h2>Dataset & Features</h2>
      <p>
        Input file: <b><code>data/features_filtered.csv</code></b> with at least the following columns:
        <code>Name</code>, <code>avg_return</code>, <code>volatility</code>, <code>skewness</code>,
        <code>kurtosis</code>, <code>mdd</code>, <code>sharpe</code>.
      </p>

      <div class="grid2">
        <div class="card">
          <h3>Why these features?</h3>
          <ul>
            <li>Return level (avg_return)</li>
            <li>Risk magnitude (volatility, mdd)</li>
            <li>Distribution shape (skewness, kurtosis)</li>
            <li>Risk-adjusted performance (sharpe)</li>
          </ul>
        </div>
        <div class="card">
          <h3>Processed output</h3>
          <ul>
            <li><code>data/processed/features_with_regime.csv</code></li>
            <li>New columns: <code>regime_id</code>, <code>regime_name</code></li>
            <li>Interpretability: clusters sorted by mean <code>avg_return</code></li>
          </ul>
        </div>
      </div>
    </section>

    <section class="section" id="pipeline">
      <h2>Pipeline (Unsupervised → Supervised)</h2>
      <div class="pipeline">
        <div class="step">
          <div class="num">1</div>
          <b>Standardize</b>
          <p>Fit <code>StandardScaler</code> on numerical features to align scales.</p>
        </div>
        <div class="step">
          <div class="num">2</div>
          <b>Cluster (KMeans)</b>
          <p>Run KMeans with <code>k=3</code>, <code>random_state=42</code>, get <code>regime_id</code>.</p>
        </div>
        <div class="step">
          <div class="num">3</div>
          <b>Name regimes</b>
          <p>Sort clusters by mean <code>avg_return</code>: Downtrend → Stable → High-Alpha.</p>
        </div>
        <div class="step">
          <div class="num">4</div>
          <b>Train classifier</b>
          <p>MLP learns to predict regime labels from 6 features for fast inference.</p>
        </div>
      </div>

      <div class="code" aria-label="Commands">
python src/create_regimes.py
python src/train.py --epochs 50 --batch_size 32 --lr 1e-3
python src/eval.py
python src/predict.py --ticker "AAPL"
      </div>
    </section>

    <section class="section" id="model">
      <h2>Model & Training Setup</h2>
      <div class="grid2">
        <div class="card">
          <h3>RegimeMLP Architecture</h3>
          <ul>
            <li>Input: 6 features</li>
            <li>Hidden: 6 → 64 (ReLU, Dropout 0.2)</li>
            <li>Hidden: 64 → 32 (ReLU, Dropout 0.2)</li>
            <li>Output: 32 → 3 logits</li>
            <li>Loss: Cross-Entropy · Optimizer: Adam</li>
          </ul>
        </div>

        <div class="card">
          <h3>Split & Checkpoints</h3>
          <ul>
            <li>Stratified split: Train ~70% / Val ~15% / Test ~15%</li>
            <li>Scaler fit on train only → <code>checkpoints/scaler.pkl</code></li>
            <li>Best model (val accuracy) → <code>checkpoints/best_model.pt</code></li>
            <li>Device: CUDA if available else CPU</li>
          </ul>
        </div>
      </div>
    </section>

    <section class="section" id="results">
      <h2>Results & Visualizations</h2>
      <p>
        After training, the evaluation script generates a classification report and plots to explain
        separability in feature space and model performance.
      </p>

      <div class="figGrid">
        <div class="figure">
          <img src="assets/pca_regimes.png" alt="PCA regimes plot"
               onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
          <div class="ph" style="display:none;">
            Put your PCA plot here:<br/>
            <code>docs/assets/pca_regimes.png</code>
          </div>
          <div class="cap">
            <b>PCA projection</b> of stocks, colored by regime. (Copy from <code>checkpoints/pca_regimes.png</code>)
          </div>
        </div>

        <div class="figure">
          <img src="assets/confusion_matrix.png" alt="Confusion matrix"
               onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
          <div class="ph" style="display:none;">
            Put your confusion matrix here:<br/>
            <code>docs/assets/confusion_matrix.png</code>
          </div>
          <div class="cap">
            <b>Confusion matrix</b> on the test set. (Copy from <code>checkpoints/confusion_matrix.png</code>)
          </div>
        </div>
      </div>

      <div class="grid2" style="margin-top:14px;">
        <div class="card">
          <h3>How to write the “impact” line (example)</h3>
          <ul>
            <li>This classifier provides a lightweight screening tool for regime-aware stock analysis.</li>
            <li>Regime labels can be used to build or compare portfolios by risk–return profiles.</li>
          </ul>
        </div>
        <div class="card">
          <h3>What to add for bonus (+1)</h3>
          <ul>
            <li>Replace XX/YY with real metrics from <code>eval.py</code></li>
            <li>Add 1–2 sentences interpreting “hardest class” and why</li>
            <li>Include the two plots above (PCA + confusion matrix)</li>
          </ul>
        </div>
      </div>
    </section>

    <section class="section" id="demo">
      <h2>Inference Demo</h2>
      <p>Predict the regime of a single ticker from the processed CSV:</p>
      <div class="code">
python src/predict.py --ticker "AAPL"

# Example (illustration)
Ticker: AAPL
Predicted regime: High-Alpha (id=2)
Probabilities: [0.05 0.10 0.85]
Description: Higher return with controlled risk.
      </div>

      <div class="grid2" style="margin-top:14px;">
        <div class="figure">
          <img src="assets/poster.png" alt="Poster preview"
               onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
          <div class="ph" style="display:none;">
            (Optional) Add poster preview image:<br/>
            <code>docs/assets/poster.png</code>
          </div>
          <div class="cap">
            Optional: upload a poster screenshot (<code>poster.png</code>) for quick preview on the webpage.
          </div>
        </div>

        <div class="card">
          <h3>Reproducibility Checklist</h3>
          <ul>
            <li>Prepare <code>data/features_filtered.csv</code></li>
            <li><code>python src/create_regimes.py</code></li>
            <li><code>python src/train.py</code></li>
            <li><code>python src/eval.py</code></li>
            <li><code>python src/predict.py --ticker "YOUR_TICKER"</code></li>
          </ul>
        </div>
      </div>
    </section>

    <section class="section" id="limits">
      <h2>Limitations & Future Work</h2>
      <div class="grid2">
        <div class="card">
          <h3>Limitations</h3>
          <ul>
            <li>Regimes come from unsupervised clustering (no external ground-truth labels).</li>
            <li>Feature window affects regime definitions (period-dependent).</li>
            <li>Only 6 summary stats: may miss tail/intraday dynamics.</li>
          </ul>
        </div>
        <div class="card">
          <h3>Future Work</h3>
          <ul>
            <li>Add features: beta, turnover, sector, valuation ratios, factor exposures.</li>
            <li>Use time-series models (CNN/LSTM) for regime transitions over time.</li>
            <li>Backtest regime-based portfolios to evaluate real investment impact.</li>
            <li>Extend to more/dynamic regimes (e.g.,
