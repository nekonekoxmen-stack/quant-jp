"""Value戦略 × 各種現金化オーバーレイの比較（ブック自身ベースを重視）。

Value純(現金化なし)が超過+4.4%/Sharpe0.74と有望だが MaxDD-44%。
TOPIXトレンド現金化は逆効果だった。ブック自身のトレンド/DD/ボラで現金化を試す。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import (
    _rebalance_dates,
    apply_regime_overlay,
    drawdown_throttle,
    run_backtest,
    topix_returns,
    trend_following_exposure,
    vol_target_exposure,
)
from quant_jp.backtest.walkforward import bootstrap_ci, deflated_sharpe, stitch_oos
from quant_jp.data import load, universe
from quant_jp.features import fundamentals as fnd
from quant_jp.features import momentum as mom

COST_BPS = float(os.environ.get("QJ_COST", "25"))

close = load.close_panel()
topix = load.load_topix().set_index("Date")["Close"]
elig = universe.eligible_mask().reindex(index=close.index, columns=close.columns).fillna(False)
bench = topix_returns(topix).reindex(close.index).fillna(0.0)
start = close.index[252]
r_b = bench.loc[start:]


def build_value_weights(top_n=20, exit_n=40, rb="Q"):
    sc = fnd.value_score(close, elig).fillna(0.5).where(elig)
    inv = (1.0 / mom.volatility(close, 63)).replace([np.inf, -np.inf], np.nan)
    days = [d for d in close.index if d in _rebalance_dates(close.index, rb)]
    held, rows = [], {}
    for d in days:
        s = sc.loc[d].dropna().sort_values(ascending=False)
        pos = {c: i + 1 for i, c in enumerate(s.index)}
        new = [c for c in held if pos.get(c, 10**9) <= exit_n]
        for c in s.index:
            if len(new) >= top_n:
                break
            if c not in new:
                new.append(c)
        held = new[:top_n]
        wv = inv.loc[d, held].replace([np.inf, -np.inf], np.nan).dropna()
        w = pd.Series(0.0, index=close.columns)
        if wv.sum() > 0:
            w.loc[wv.index] = wv / wv.sum()
        rows[d] = w
    return pd.DataFrame(rows).T.reindex(index=close.index, columns=close.columns).ffill().fillna(0.0)


w = build_value_weights()
res = run_backtest(close, w, exposure=None, rebalance="Q", cost_bps=COST_BPS)
base = res.returns

# ブック自身の equity からトレンド/DDを作る
book_eq = (1 + base.fillna(0)).cumprod()


def book_trend_exposure(ma=200, low=0.3):
    ma_s = book_eq.rolling(ma, min_periods=ma // 2).mean()
    sig = (book_eq > ma_s).astype(float)
    return (low + (1 - low) * sig).shift(1).fillna(1.0)


overlays = {
    "現金化なし": pd.Series(1.0, index=base.index),
    "ブックDDスロットル": drawdown_throttle(base, soft=-0.12, hard=-0.28),
    "ブックボラ目標0.15": vol_target_exposure(base, target_vol=0.15),
    "ブック200日トレンド": book_trend_exposure(200, 0.3),
    "TOPIXトレンド(参考)": trend_following_exposure(topix, base.index),
}

print(f"== Value戦略 × 現金化（コスト{COST_BPS:.0f}bps, 回転率93%程度） ==")
print("オーバーレイ | CAGR | Sharpe | MaxDD | 超過 | 年次勝率")
print("---|---|---|---|---|---")
saved = {}
for name, exp in overlays.items():
    r = apply_regime_overlay(base, exp, cost_bps=COST_BPS).loc[start:]
    saved[name] = r
    s = metrics.summary(r, benchmark=r_b)
    a_s = (1 + r).groupby(r.index.year).prod() - 1
    a_b = (1 + r_b).groupby(r_b.index.year).prod() - 1
    win = int((a_s - a_b > 0).sum())
    print(f"{name} | {s['CAGR']:.2%} | {s['Sharpe']:.2f} | {s['MaxDD']:.2%} | {s['Excess_CAGR']:.2%} | {win}/{len(a_s)}")

print(f"\nTOPIX | {metrics.cagr(r_b):.2%} | {metrics.sharpe(r_b):.2f} | {metrics.max_drawdown(r_b):.2%}")

# 最良で有意性
for cand in ["ブックボラ目標0.15", "ブックDDスロットル", "ブック200日トレンド"]:
    r = saved[cand]
    ci = bootstrap_ci(r, r_b)
    print(f"\n[{cand}] 超過CAGR {ci['excess_cagr_mean']:.2%} 95%CI[{ci['excess_cagr_ci'][0]:.2%},{ci['excess_cagr_ci'][1]:.2%}] "
          f"p {ci['excess_cagr_pval']:.3f} / DSR {deflated_sharpe(r, n_trials=15):.3f}")
    oos = stitch_oos(r, close.index, train_years=3, test_years=1)
    so = metrics.summary(oos, benchmark=r_b.reindex(oos.index))
    print(f"  OOS: CAGR {so['CAGR']:.2%} Sharpe {so['Sharpe']:.2f} 超過 {so['Excess_CAGR']:.2%}")
