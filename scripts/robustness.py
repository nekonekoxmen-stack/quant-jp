"""頑健性チェック: 年次の対TOPIX一貫性と、現金化(エクスポージャー)の作動状況。"""

from __future__ import annotations

import numpy as np
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

COST_BPS, REBAL, TARGET_VOL = 10.0, "ME", 0.15

close = load.close_panel()
topix = load.load_topix().set_index("Date")["Close"]
eligible = universe.eligible_mask().reindex(index=close.index, columns=close.columns).fillna(False)
weights = ranking.select_weights(close, eligible, top_n=15)
res_inv = run_backtest(close, weights, exposure=None, rebalance=REBAL, cost_bps=COST_BPS)
exposure = vol_target_exposure(res_inv.returns, target_vol=TARGET_VOL)
strat = apply_regime_overlay(res_inv.returns, exposure, cost_bps=COST_BPS)
bench = topix_returns(topix).reindex(close.index).fillna(0.0)

start = close.index[252]
strat, bench, exposure = strat.loc[start:], bench.loc[start:], exposure.loc[start:]

# 年次リターン比較
def annual(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1

a_s, a_b = annual(strat), annual(bench)
print("年 | 戦略 | TOPIX | 超過 | 平均エクスポージャー")
print("---|---|---|---|---")
win = 0
for y in a_s.index:
    ex = a_s[y] - a_b[y]
    win += ex > 0
    avg_exp = exposure[exposure.index.year == y].mean()
    print(f"{y} | {a_s[y]:.1%} | {a_b[y]:.1%} | {ex:+.1%} | {avg_exp:.0%}")
print(f"\n勝率(年次で対TOPIX): {win}/{len(a_s)}")

# 主要ストレス局面のエクスポージャー（現金化が効いたか）
print("\n現金化チェック（局面別の平均株式エクスポージャー）:")
for label, lo, hi in [
    ("2018Q4 急落", "2018-10-01", "2018-12-31"),
    ("2020 コロナ", "2020-02-20", "2020-04-30"),
    ("2022 下落", "2022-01-01", "2022-06-30"),
]:
    seg = exposure.loc[lo:hi]
    print(f"  {label}: 平均 {seg.mean():.0%} / 最低 {seg.min():.0%}")

print("\n全期間エクスポージャー: 平均 {:.0%} / 中央 {:.0%}".format(exposure.mean(), exposure.median()))
print("metrics:", metrics.format_summary(metrics.summary(strat, benchmark=bench)))
