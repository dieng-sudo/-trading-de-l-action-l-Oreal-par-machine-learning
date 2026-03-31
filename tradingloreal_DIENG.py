import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# Paramètres
# =========================
CSV_PATH = "LOREAL2.csv"   # <-- fichier corrigé
COST_PER_FLIP = 0.000      # 10 bps par inversion long<->short
FREQ = 252                 # annualisation (données journalières)
TRAIN_RATIO = 0.70         # split chronologique 

# =========================
# 1) Load data
# =========================
df = pd.read_csv(CSV_PATH, parse_dates=["Date"]).sort_values("Date").set_index("Date")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"])
close = df["Close"]

# =========================
# 2) Objectif: +1 si le lendemain monte, sinon -1
# =========================
df_ml = df.copy()
df_ml["objectif"] = np.where(close.shift(-1) > close, 1, -1)

# =========================
# 3) Features demandées
# =========================
df_ml["close_prev"] = close.shift(1)
df_ml["close_today"] = close
df_ml["sma10"] = close.rolling(10).mean()
df_ml["sma20"] = close.rolling(20).mean()
df_ml["sma50"] = close.rolling(50).mean()

df_ml = df_ml.dropna(subset=["objectif", "close_prev", "close_today", "sma10", "sma20", "sma50"])

X = df_ml[["close_prev", "close_today", "sma10", "sma20", "sma50"]]
y = df_ml["objectif"]

# =========================
# 4) Split chronologique 80/20
# =========================
split_idx = int(len(df_ml) * TRAIN_RATIO)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_index = X_test.index

# =========================
# 5) Rendements d'intervalle (t -> t+1 indexés par t)
# =========================
interval_ret = close.pct_change().shift(-1)
interval_ret_test = interval_ret.reindex(test_index)

def compute_ann_return_and_flips(interval_ret: pd.Series,
                                 pos: pd.Series,
                                 cost_per_flip: float = COST_PER_FLIP,
                                 freq: int = FREQ):
    pos = pos.reindex(interval_ret.index).ffill().fillna(1)
    flip = (pos.diff().abs() / 2).fillna(0)
    net_ret = (pos * interval_ret).fillna(0) - cost_per_flip * flip

    daily = net_ret.replace([np.inf, -np.inf], np.nan).dropna()
    if len(daily) > 1:
        ann_return_net = (1 + daily).prod() ** (freq / len(daily)) - 1
    else:
        ann_return_net = np.nan

    return float(ann_return_net) if ann_return_net == ann_return_net else np.nan, int(flip.sum())

# =========================
# 6) Baselines A–E (signaux à t utilisés pour t->t+1)
# =========================
def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

signals_baseline = {
    "A_sma20_trend": pd.Series(np.where(close > close.rolling(20).mean(), 1, -1), index=close.index),
    "B_ma20_50_cross": pd.Series(np.where(close.rolling(20).mean() > close.rolling(50).mean(), 1, -1), index=close.index),
    "C_momentum10": pd.Series(np.where((close / close.shift(10) - 1) > 0, 1, -1), index=close.index),
    "D_rsi14_50": pd.Series(np.where(rsi_wilder(close, 14) >= 50, 1, -1), index=close.index),
    "E_zscore20_revert": pd.Series(np.where(zscore(close, 20) < 0, 1, -1), index=close.index),
}

# =========================
# 7) Modèles ML (Random Forest allégée pour vitesse)
# =========================
models = {
    "ML1_random_forest": RandomForestClassifier(
        n_estimators=500, max_depth=8, random_state=42, n_jobs=1
    ),
    "ML2_logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ]),
    "ML3_knn": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=50))
    ]),
}

# =========================
# 8) Évaluation sur TEST + tableau synthèse
# =========================
rows = []
test_positions = {}
test_objectif = y_test.copy()

# Baselines
for name, sig in signals_baseline.items():
    pos_test = sig.reindex(test_index)
    ann_ret, flips = compute_ann_return_and_flips(interval_ret_test, pos_test)
    rows.append({"strategy": name, "ann_return_net": ann_ret, "flips": flips})
    test_positions[name] = pos_test

# ML models
for name, model in models.items():
    model.fit(X_train, y_train)
    pred_test = pd.Series(model.predict(X_test), index=test_index)  # ±1
    ann_ret, flips = compute_ann_return_and_flips(interval_ret_test, pred_test)
    rows.append({"strategy": name, "ann_return_net": ann_ret, "flips": flips})
    test_positions[name] = pred_test

results = pd.DataFrame(rows).sort_values("ann_return_net", ascending=False)
print(results.to_string(index=False))

# =========================
# 9) Graphique 3 sous-graphes pour la meilleure méthode
# =========================
best_name = results.iloc[0]["strategy"]
best_pred = test_positions[best_name].reindex(test_index).astype(int)
test_close = close.reindex(test_index)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# (1) Close + prédictions
ax1.plot(test_index, test_close, label="Close (test)")
ax1.scatter(test_index[best_pred == 1], test_close[best_pred == 1], marker="^", label="Prédiction +1")
ax1.scatter(test_index[best_pred == -1], test_close[best_pred == -1], marker="v", label="Prédiction -1")
ax1.set_title(f"{best_name} — Test : Prédictions (haut), Objectif (milieu), Bande de classes (bas)")
ax1.set_ylabel("Close")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.2)

# (2) Close + objectif
ax2.plot(test_index, test_close, label="Close (test)")
ax2.scatter(test_index[test_objectif == 1], test_close[test_objectif == 1], marker="o", label="Objectif +1")
ax2.scatter(test_index[test_objectif == -1], test_close[test_objectif == -1], marker="x", label="Objectif -1")
ax2.set_ylabel("Close")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.2)

# (3) Bande de classes : prédiction vs objectif (+1/-1)
ax3.axhline(0, linewidth=1)
ax3.step(test_index, best_pred.values, where="post", label="Prédiction (±1)")
ax3.step(test_index, test_objectif.values, where="post", linestyle="--", label="Objectif (±1)")

mismatch = (best_pred != test_objectif)
if mismatch.any():
    ax3.scatter(test_index[mismatch], best_pred[mismatch], marker="s", label="Mismatch (pred != obj)")

ax3.set_yticks([-1, 1])
ax3.set_ylabel("Classe")
ax3.set_xlabel("Date")
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
