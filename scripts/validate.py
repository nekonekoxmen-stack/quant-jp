"""監査後の検証: 回転率(バッファ有無), コスト感度, IS/OOS分割。"""

from __future__ import annotations

import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import apply_regime_overlay, run_backtest, topix_returns, vol_target_exposure
from quant_jp.data import load, universe
from quant_jp.strategy import ranking

TARGET_VOL, REBAL = 0.15, "ME"

close = load.close_panel()
topix = load.load_topix().set_index("Date")["Close"]
eligible = universe.eligible_mask().reindex(index=close.index, columns=close.columns).fillna(False)
bench = topix_returns(topix).reindex(close.index).fillna(0.0)
start = close.index[252]


def run(weights, cost_bps):
    res = run_backtest(close, weights, exposure=None, rebalance=REBAL, cost_bps=cost_bps)
    exp = vol_target_exposure(res.returns, target_vol=TARGET_VOL)
    r = apply_regime_overlay(res.returns, exp, cost_bps=cost_bps)
    ann_turn = res.turnover.loc[start:].sum() / (len(res.turnover.loc[start:]) / 252)
    return r, ann_turn


w_plain = ranking.select_weights(close, eligible, top_n=15)
w_buf = ranking.select_weights_buffered(close, eligible, top_n=15, exit_n=30, rebalance=REBAL)

print("== 回転率（年率, 片道）バッファ効果 ==")
for name, w in [("バッファなし", w_plain), ("バッファあり(exit30)", w_buf)]:
    r, t = run(w, 10.0)
    s = metrics.summary(r.loc[start:], benchmark=bench.loc[start:])
    print(f"{name}: 回転率 {t:.0%} | CAGR {s['CAGR']:.2%} | Sharpe {s['Sharpe']:.2f} | "
          f"MaxDD {s['MaxDD']:.2%} | 超過 {s['Excess_CAGR']:.2%}")

print("\n== コスト感度（バッファあり） ==")
for cb in [10.0, 20.0, 30.0, 50.0]:
    r, _ = run(w_buf, cb)
    s = metrics.summary(r.loc[start:], benchmark=bench.loc[start:])
    print(f"cost={cb:.0f}bps: CAGR {s['CAGR']:.2%} | Sharpe {s['Sharpe']:.2f} | 超過 {s['Excess_CAGR']:.2%}")

print("\n== IS/OOS 分割（バッファあり, 10bps） ==")
r_buf, _ = run(w_buf, 10.0)
for name, lo, hi in [("IS 2017-2021", "2017-06-07", "2021-12-31"), ("OOS 2022-2026", "2022-01-01", "2026-05-29")]:
    rr = r_buf.loc[lo:hi]
    bb = bench.loc[lo:hi]
    s = metrics.summary(rr, benchmark=bb)
    print(f"{name}: CAGR {s['CAGR']:.2%} | Sharpe {s['Sharpe']:.2f} | MaxDD {s['MaxDD']:.2%} | "
          f"TOPIX {s['Bench_CAGR']:.2%} | 超過 {s['Excess_CAGR']:.2%}")
