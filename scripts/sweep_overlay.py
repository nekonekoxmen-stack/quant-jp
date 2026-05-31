"""リスク・オーバーレイのスイープ（データ読込は1回）。

マルチファクター・ブックに対し、ボラ目標と弱気ゲートの掛け方を変えて
対TOPIXの CAGR / Sharpe / MaxDD を比較する。
"""

from __future__ import annotations

import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import (
    apply_regime_overlay,
    drawdown_throttle,
    run_backtest,
    topix_returns,
    vol_target_exposure,
)
from quant_jp.data import load, universe
from quant_jp.features import trend
from quant_jp.strategy import ranking
from quant_jp.strategy.regime_overlay import market_exposure

COST_BPS = 10.0
REBAL = "ME"

print("読込...", flush=True)
close = load.close_panel()
topix = load.load_topix().set_index("Date")["Close"]
eligible = universe.eligible_mask().reindex(index=close.index, columns=close.columns).fillna(False)
weights = ranking.select_weights(close, eligible, top_n=15)
breadth = trend.breadth_above_sma(close, 200, eligible)
reg = market_exposure(topix, breadth, target_vol=1e9, max_vol_leverage=1.0)
bear_gate = (reg["trend"] * reg["dd_factor"] * reg["breadth_factor"]).clip(0.0, 1.0)

res_inv = run_backtest(close, weights, exposure=None, rebalance=REBAL, cost_bps=COST_BPS)
bench = topix_returns(topix).reindex(close.index).fillna(0.0)
start = close.index[252]
r_bench = bench.loc[start:]

rows = []


def add(name: str, r: pd.Series) -> None:
    s = metrics.summary(r.loc[start:], benchmark=r_bench)
    rows.append((name, s))


add("フル投資(制御なし)", res_inv.returns)

ddt = drawdown_throttle(res_inv.returns, soft=-0.10, hard=-0.20)
for tv in [0.14, 0.16, 0.18, 0.20, 0.25]:
    vt = vol_target_exposure(res_inv.returns, target_vol=tv)
    # ボラ目標のみ
    add(f"voltgt={tv:.2f}", apply_regime_overlay(res_inv.returns, vt, cost_bps=COST_BPS))
    # ボラ目標 × ブックDDスロットル
    exp = (vt * ddt).clip(0.0, 1.0)
    add(f"voltgt={tv:.2f}×ddthr", apply_regime_overlay(res_inv.returns, exp, cost_bps=COST_BPS))

add("TOPIX", bench)

print(f"\n評価期間: {start.date()} 〜 {close.index[-1].date()}\n")
hdr = ["戦略", "CAGR", "Sharpe", "MaxDD", "超過CAGR"]
print(" | ".join(hdr))
print("---|" * len(hdr))
for name, s in rows:
    ex = s.get("Excess_CAGR", float("nan"))
    print(f"{name} | {s['CAGR']:.2%} | {s['Sharpe']:.2f} | {s['MaxDD']:.2%} | {ex:.2%}")
