"""監査・Gemini提言の改善を検証: TTM-EPS後のValueに quality結合/セクター中立を重ねる。

各案を 25bps・現金化あり・OOS・ブートストラップで比較し、過剰最適化でない改善のみ採用。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import (
    _rebalance_dates,
    apply_regime_overlay,
    book_trend_exposure,
    run_backtest,
    topix_returns,
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
sectors = universe.sector_map(close.columns)


def score_value(quality_w=0.0, sector_neutral=False):
    p = fnd.fundamental_panels(close)
    def rk(panel):
        if sector_neutral:
            return mom.sector_neutral_rank(panel, sectors, elig)
        return mom.cross_sectional_rank(panel, elig)
    val = (rk(p["EP"]) + rk(p["BP"])) / 2
    if quality_w <= 0:
        return val.fillna(0.5)
    qual = (rk(p["OP_TA"]) + rk(p["ROE"]) + rk(p["EqAR"])) / 3
    return ((1 - quality_w) * val.fillna(0.5) + quality_w * qual.fillna(0.5))


def build(score, top_n=20, exit_n=40, rb="Q"):
    sc = score.where(elig)
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
        iv = inv.loc[d, held].replace([np.inf, -np.inf], np.nan).dropna()
        w = pd.Series(0.0, index=close.columns)
        if iv.sum() > 0:
            wv = iv / iv.sum()
            cap = max(2.0 / max(len(held), 1), 0.10)
            for _ in range(3):
                over = wv > cap
                if not over.any():
                    break
                exc = (wv[over] - cap).sum()
                wv[over] = cap
                und = ~over
                if wv[und].sum() > 0:
                    wv[und] += exc * wv[und] / wv[und].sum()
            w.loc[wv.index] = wv
        rows[d] = w
    return pd.DataFrame(rows).T.reindex(index=close.index, columns=close.columns).ffill().fillna(0.0)


def evaluate(score):
    w = build(score)
    res = run_backtest(close, w, exposure=None, rebalance="Q", cost_bps=COST_BPS)
    exp = book_trend_exposure(res.returns, ma_window=200, low=0.30)
    r = apply_regime_overlay(res.returns, exp, cost_bps=COST_BPS).loc[start:]
    turn = res.turnover.loc[start:].sum() / (len(res.turnover.loc[start:]) / 252)
    return r, turn


print(f"== 改善案の比較（コスト{COST_BPS:.0f}bps, TTM-EPS, 現金化あり） ==")
print("構成 | CAGR | Sharpe | MaxDD | 超過 | 勝率 | 回転率")
print("---|---|---|---|---|---|---")
cases = {
    "Value純": score_value(0.0, False),
    "Value+Q0.3": score_value(0.3, False),
    "Value+Q0.5": score_value(0.5, False),
    "Valueセクター中立": score_value(0.0, True),
    "V+Q0.3セクター中立": score_value(0.3, True),
}
saved = {}
for name, sc in cases.items():
    r, turn = evaluate(sc)
    saved[name] = r
    s = metrics.summary(r, benchmark=r_b)
    a_s = (1 + r).groupby(r.index.year).prod() - 1
    a_b = (1 + r_b).groupby(r_b.index.year).prod() - 1
    win = int((a_s - a_b > 0).sum())
    print(f"{name} | {s['CAGR']:.2%} | {s['Sharpe']:.2f} | {s['MaxDD']:.2%} | {s['Excess_CAGR']:.2%} | {win}/{len(a_s)} | {turn:.0%}")

print(f"\nTOPIX | {metrics.cagr(r_b):.2%} | {metrics.sharpe(r_b):.2f} | {metrics.max_drawdown(r_b):.2%}")

for cand in saved:
    r = saved[cand]
    ci = bootstrap_ci(r, r_b)
    oos = stitch_oos(r, close.index, train_years=3, test_years=1)
    so = metrics.summary(oos, benchmark=r_b.reindex(oos.index))
    print(f"[{cand}] p {ci['excess_cagr_pval']:.3f} / DSR {deflated_sharpe(r, n_trials=15):.3f} "
          f"/ OOS超過 {so['Excess_CAGR']:.2%}(Sh {so['Sharpe']:.2f})")
