# 📈 L'Oréal Stock Direction Classifier

> A machine learning pipeline that predicts next-day price direction (long/short) for L'Oréal (OR.PA) and benchmarks it against five classical technical analysis strategies.

---

## 🧠 Overview

This project builds a **binary directional classifier** (`+1` = up, `-1` = down) on L'Oréal daily close prices, comparing three ML models against five rule-based baselines. The evaluation metric is **net annualised return** on a held-out chronological test set (last 30% of data), including transaction costs.

---

## 📁 Project Structure

```
.
├── LOREAL2.csv          # Input data — daily OHLCV for L'Oréal
├── loreal_classifier.py # Main script
└── README.md
```

---

## ⚙️ Methodology

### Target Variable
```
objectif[t] = +1  if Close[t+1] > Close[t]
            = -1  otherwise
```
The model predicts tomorrow's direction using today's information only — no lookahead.

### Features
| Feature | Description |
|---|---|
| `close_prev` | Yesterday's closing price |
| `close_today` | Today's closing price |
| `sma10` | 10-day simple moving average |
| `sma20` | 20-day simple moving average |
| `sma50` | 50-day simple moving average |

### Train / Test Split
Chronological split — **no shuffling** to prevent data leakage:
- **Train**: first 70% of observations
- **Test**: last 30% of observations

---

## 📊 Strategies Compared

### Baselines (rule-based signals)

| ID | Strategy | Logic |
|---|---|---|
| A | `sma20_trend` | Long if Close > SMA(20) |
| B | `ma20_50_cross` | Long if SMA(20) > SMA(50) |
| C | `momentum10` | Long if 10-day return > 0 |
| D | `rsi14_50` | Long if RSI(14) ≥ 50 |
| E | `zscore20_revert` | Long if Z-score(20) < 0 (mean reversion) |

### ML Models

| ID | Model | Notes |
|---|---|---|
| ML1 | Random Forest | 500 trees, max depth 8 |
| ML2 | Logistic Regression | L-BFGS solver, StandardScaler |
| ML3 | K-Nearest Neighbours | k=50, StandardScaler |

---

## 💰 Performance Metric

Each strategy is evaluated by its **net annualised return** on the test set:

```
net_return[t] = position[t] × interval_return[t] − cost_per_flip × |Δposition[t]| / 2
```

Annualised using the geometric compounding formula over 252 trading days.

> **Default transaction cost**: `COST_PER_FLIP = 0.000` (0 bps). Set to e.g. `0.001` (10 bps) to model realistic slippage.

---

## 📈 Output

### Console — Rankings Table
```
strategy               ann_return_net  flips
ML1_random_forest            0.1842     312
A_sma20_trend                0.1105      18
...
```

### Plots — 3-panel chart for the best strategy
1. **Close price + predicted signals** (▲ long / ▼ short)
2. **Close price + ground-truth labels**
3. **Prediction vs. objective timeline** with mismatch markers

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2. Add your data

Place your CSV file at the path defined by `CSV_PATH` (default: `LOREAL2.csv`).

Expected columns:

| Column | Format |
|---|---|
| `Date` | Parseable date string (e.g. `2020-01-15`) |
| `Close` | Numeric closing price |

### 3. Run

```bash
python loreal_classifier.py
```

---

## 🔧 Configuration

All key parameters are at the top of the script:

```python
CSV_PATH      = "LOREAL2.csv"  # Path to your data file
COST_PER_FLIP = 0.000          # Transaction cost per position flip (e.g. 0.001 = 10 bps)
FREQ          = 252            # Trading days per year (annualisation factor)
TRAIN_RATIO   = 0.70           # Fraction of data used for training
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice. Past backtested performance is not indicative of future results. Always validate on out-of-sample data before drawing conclusions.

---

## 📄 License

MIT
