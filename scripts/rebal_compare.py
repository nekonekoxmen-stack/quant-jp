"""銘柄入れ替え頻度の比較（週次/月次/四半期）。現金化(ボラ目標)は日次で共通適用。"""

from __future__ import annotations

import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import (
    apply_regime_overlay,
    run_backtest,
    topix_returns,
    vol_target_exposure,
)
from quant_jp.data import load, universe
from quant_jp.strategy import ranking

COST_BPS, TARGET_VOL = 10.0, 0.15

close = load.close_panel()
topix = load.load_topix().set_index("Date")["Close"]
eligible = universe.eligible_mask().reindex(index=close.index, columns=close.columns).fillna(False)
weights = ranking.select_weights(close, eligible, top_n=15)
bench = topix_returns(topix).reindex(close.index).fillna(0.0)
start = close.index[252]
r_b = bench.loc[start:]

print("入れ替え頻度 | CAGR | Sharpe | MaxDD | 超過 | 年間回転率")
print("---|---|---|---|---|---")
for label, freq in [("週次", "W-FRI"), ("月次", "ME"), ("四半期", "Q")]:
    res = run_backtest(close, weights, exposure=None, rebalance=freq, cost_bps=COST_BPS)
    exp = vol_target_exposure(res.returns, target_vol=TARGET_VOL)
    r = apply_regime_overlay(res.returns, exp, cost_bps=COST_BPS).loc[start:]
    s = metrics.summary(r, benchmark=r_b)
    # 片道回転率の年率換算（営業日252で規格化）
    ann_turn = res.turnover.loc[start:].sum() / (len(res.turnover.loc[start:]) / 252)
    print(
        f"{label}({freq}) | {s['CAGR']:.2%} | {s['Sharpe']:.2f} | {s['MaxDD']:.2%} | "
        f"{s['Excess_CAGR']:.2%} | {ann_turn:.0%}"
    )
print(f"\n（参考）TOPIX CAGR {metrics.cagr(r_b):.2%} / Sharpe {metrics.sharpe(r_b):.2f}")
