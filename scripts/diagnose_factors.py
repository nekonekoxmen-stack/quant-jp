"""各ファクター単体の予測力をコスト0で確認（四半期, 上位20等ウェイト）。"""

from __future__ import annotations

import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import _rebalance_dates, run_backtest, topix_returns, trend_following_exposure
from quant_jp.data import load, universe
from quant_jp.features import fundamentals as fnd
from quant_jp.strategy import ranking

close = load.close_panel()
topix = load.load_topix().set_index("Date")["Close"]
elig = universe.eligible_mask().reindex(index=close.index, columns=close.columns).fillna(False)
bench = topix_returns(topix).reindex(close.index).fillna(0.0)
start = close.index[252]
r_b = bench.loc[start:]


def topw(score, top=20, rb="Q"):
    sc = score.where(elig)
    days = [d for d in close.index if d in _rebalance_dates(close.index, rb)]
    rows = {}
    for d in days:
        s = sc.loc[d].dropna().sort_values(ascending=False).head(top)
        w = pd.Series(0.0, index=close.columns)
        if len(s) > 0:
            w.loc[s.index] = 1.0 / len(s)
        rows[d] = w
    return pd.DataFrame(rows).T.reindex(index=close.index, columns=close.columns).ffill().fillna(0.0)


factors = [
    ("Value", fnd.value_score(close, elig)),
    ("Quality", fnd.quality_score(close, elig)),
    ("LowVol", fnd.low_volatility_score(close, elig)),
    ("合成LowTurn", ranking.lowturn_score(close, elig)),
]
print("ファクター単体（コスト0, 四半期, 上位20等ウェイト）")
print("名前 | CAGR | Sharpe | MaxDD | 超過")
for name, sc in factors:
    w = topw(sc)
    res = run_backtest(close, w, exposure=None, rebalance="Q", cost_bps=0.0)
    s = metrics.summary(res.returns.loc[start:], benchmark=r_b)
    print(f"{name} | {s['CAGR']:.2%} | {s['Sharpe']:.2f} | {s['MaxDD']:.2%} | {s['Excess_CAGR']:.2%}")

print(f"\nTOPIX | {metrics.cagr(r_b):.2%} | {metrics.sharpe(r_b):.2f} | {metrics.max_drawdown(r_b):.2%}")

exp = trend_following_exposure(topix, close.index).loc[start:]
print(f"\nトレンド現金化: 平均エクスポージャー {exp.mean():.0%} / 最低 {exp.min():.0%}")
# 年ごとのエクスポージャー
ann = exp.groupby(exp.index.year).mean()
print("年別平均エクスポージャー:", {int(y): round(float(v), 2) for y, v in ann.items()})
