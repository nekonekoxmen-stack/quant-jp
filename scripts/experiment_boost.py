"""「もう少し上」を狙う実験: バリュー×モメンタムのブレンド と 値がさ株フィルタ緩和。

単元未満株(1株単位)運用なら値がさ株を除外する必要はない → 半導体等の高価格・
高モメンタム株を拾える。ブレンド比率と値がさ上限を変えてネット成績を比較する。
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
bench = topix_returns(topix).reindex(close.index).fillna(0.0)
start = close.index[252]
r_b = bench.loc[start:]


def make_elig(max_lot):
    e = universe.eligible_mask(max_lot_cost_yen=max_lot)
    return e.reindex(index=close.index, columns=close.columns).fillna(False)


def blended_score(elig, mom_w):
    """バリュー(1-mom_w) + モメンタム(mom_w) の横断ランク合成。"""
    v = fnd.value_score(close, elig).fillna(0.5)
    if mom_w <= 0:
        return v
    m = mom.cross_sectional_rank(mom.momentum_12_1(close), elig)
    m6 = mom.cross_sectional_rank(mom.momentum_6_1(close), elig)
    mscore = (m + m6) / 2
    return (1 - mom_w) * v + mom_w * mscore.fillna(0.5)


def build(score, elig, top_n=20, exit_n=40, rb="Q"):
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
            w.loc[iv.index] = iv / iv.sum()
        rows[d] = w
    return pd.DataFrame(rows).T.reindex(index=close.index, columns=close.columns).ffill().fillna(0.0)


def evaluate(score, elig, cashout=True):
    w = build(score, elig)
    res = run_backtest(close, w, exposure=None, rebalance="Q", cost_bps=COST_BPS)
    if cashout:
        exp = book_trend_exposure(res.returns, ma_window=200, low=0.30)
        r = apply_regime_overlay(res.returns, exp, cost_bps=COST_BPS)
    else:
        r = res.returns
    turn = res.turnover.loc[start:].sum() / (len(res.turnover.loc[start:]) / 252)
    return r.loc[start:], turn


print(f"== ブレンド & 値がさ解禁の実験（コスト{COST_BPS:.0f}bps, 現金化あり） ==")
print("構成 | CAGR | Sharpe | MaxDD | 超過 | 勝率 | 回転率")
print("---|---|---|---|---|---|---")

elig_strict = make_elig(1e6)     # 現行: 1単元100万円以下
elig_loose = make_elig(1e9)      # 値がさ解禁（単元未満株前提）

saved = {}
for label, elig, mom_w in [
    ("Value純/値がさ制限", elig_strict, 0.0),
    ("Value純/値がさ解禁", elig_loose, 0.0),
    ("Value+Mom0.3/解禁", elig_loose, 0.3),
    ("Value+Mom0.5/解禁", elig_loose, 0.5),
    ("Mom主軸0.7/解禁", elig_loose, 0.7),
]:
    sc = blended_score(elig, mom_w)
    r, turn = evaluate(sc, elig)
    saved[label] = r
    s = metrics.summary(r, benchmark=r_b)
    a_s = (1 + r).groupby(r.index.year).prod() - 1
    a_b = (1 + r_b).groupby(r_b.index.year).prod() - 1
    win = int((a_s - a_b > 0).sum())
    print(f"{label} | {s['CAGR']:.2%} | {s['Sharpe']:.2f} | {s['MaxDD']:.2%} | "
          f"{s['Excess_CAGR']:.2%} | {win}/{len(a_s)} | {turn:.0%}")

print(f"\nTOPIX | {metrics.cagr(r_b):.2%} | {metrics.sharpe(r_b):.2f} | {metrics.max_drawdown(r_b):.2%}")

# 最良候補の有意性
for cand in ["Value+Mom0.3/解禁", "Value純/値がさ解禁"]:
    r = saved[cand]
    ci = bootstrap_ci(r, r_b)
    oos = stitch_oos(r, close.index, train_years=3, test_years=1)
    so = metrics.summary(oos, benchmark=r_b.reindex(oos.index))
    print(f"\n[{cand}] 超過CAGR p値 {ci['excess_cagr_pval']:.3f} / DSR {deflated_sharpe(r, n_trials=15):.3f} "
          f"/ OOS超過 {so['Excess_CAGR']:.2%}(Sharpe {so['Sharpe']:.2f})")
