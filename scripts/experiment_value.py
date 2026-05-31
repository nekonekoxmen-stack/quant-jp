"""Value中心戦略の検証（コスト25bps, 上場廃止処理, トレンド現金化）。

診断で Value が突出して強い（超過+6.6%, コスト0）と判明。これを主軸に、
Quality を「バリュー罠の足切りフィルタ」として軽く併用し、低ボラ加重・四半期
バッファ運用・トレンドフォロー現金化を重ねて、ネット/有意性まで評価する。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import (
    _rebalance_dates,
    apply_regime_overlay,
    run_backtest,
    topix_returns,
    trend_following_exposure,
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


def value_quality_score(quality_w: float) -> pd.DataFrame:
    """Value 主軸 + Quality を軽く加味（バリュー罠除け）。"""
    v = fnd.value_score(close, elig).fillna(0.5)
    q = fnd.quality_score(close, elig).fillna(0.5)
    return (1 - quality_w) * v + quality_w * q


def build_weights(score, top_n=20, exit_n=40, rb="Q", inv_vol=True):
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
        if inv_vol:
            wv = inv.loc[d, held].replace([np.inf, -np.inf], np.nan).dropna()
        else:
            wv = pd.Series(1.0, index=held)
        w = pd.Series(0.0, index=close.columns)
        if wv.sum() > 0:
            w.loc[wv.index] = wv / wv.sum()
        rows[d] = w
    return pd.DataFrame(rows).T.reindex(index=close.index, columns=close.columns).ffill().fillna(0.0)


def run(score, overlay="trend", **kw):
    w = build_weights(score, **kw)
    res = run_backtest(close, w, exposure=None, rebalance=kw.get("rb", "Q"), cost_bps=COST_BPS)
    if overlay == "trend":
        exp = trend_following_exposure(topix, res.returns.index)
        r = apply_regime_overlay(res.returns, exp, cost_bps=COST_BPS)
    else:
        r = res.returns
    turn = res.turnover.loc[start:].sum() / (len(res.turnover.loc[start:]) / 252)
    return r, turn


print(f"== Value中心戦略（コスト{COST_BPS:.0f}bps） ==")
print("構成 | CAGR | Sharpe | MaxDD | 超過 | 年次勝率 | 回転率")
print("---|---|---|---|---|---|---")

variants = {
    "Value純, 現金化なし": (value_quality_score(0.0), "none"),
    "Value純, トレンド現金化": (value_quality_score(0.0), "trend"),
    "Value+Q0.2, トレンド": (value_quality_score(0.2), "trend"),
    "Value+Q0.3, トレンド": (value_quality_score(0.3), "trend"),
}
saved = {}
for name, (sc, ov) in variants.items():
    r, turn = run(sc, overlay=ov, top_n=20, exit_n=40, rb="Q")
    rr = r.loc[start:]
    saved[name] = rr
    s = metrics.summary(rr, benchmark=r_b)
    a_s = (1 + rr).groupby(rr.index.year).prod() - 1
    a_b = (1 + r_b).groupby(r_b.index.year).prod() - 1
    win = int((a_s - a_b > 0).sum())
    print(f"{name} | {s['CAGR']:.2%} | {s['Sharpe']:.2f} | {s['MaxDD']:.2%} | "
          f"{s['Excess_CAGR']:.2%} | {win}/{len(a_s)} | {turn:.0%}")

print(f"\nTOPIX | {metrics.cagr(r_b):.2%} | {metrics.sharpe(r_b):.2f} | {metrics.max_drawdown(r_b):.2%}")

best = saved["Value+Q0.2, トレンド"]
print("\n== Walk-forward OOS ==")
oos = stitch_oos(best, close.index, train_years=3, test_years=1)
s = metrics.summary(oos, benchmark=r_b.reindex(oos.index))
print(f"OOS: CAGR {s['CAGR']:.2%} | Sharpe {s['Sharpe']:.2f} | MaxDD {s['MaxDD']:.2%} | 超過 {s['Excess_CAGR']:.2%}")

print("\n== ブートストラップ有意性 ==")
ci = bootstrap_ci(best, r_b)
print(f"超過CAGR平均 {ci['excess_cagr_mean']:.2%}, 95%CI [{ci['excess_cagr_ci'][0]:.2%}, {ci['excess_cagr_ci'][1]:.2%}], p値 {ci['excess_cagr_pval']:.3f}")
print(f"Sharpe差 p値 {ci['sharpe_diff_pval']:.3f} / Deflated Sharpe {deflated_sharpe(best, n_trials=12):.3f}")
