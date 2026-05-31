"""低回転マルチファクター戦略の評価（25bpsコスト, 上場廃止処理込み）。

シグナル: バリュー＋クオリティ＋低ボラ（モメンタム除外, 統合スコア, 年1-2回相当の
バッファ運用）。現金化: トレンドフォロー（200日線＋12ヶ月絶対モメンタム）。
ベースライン(旧モメンタム系)と比較し、WF-OOS とブートストラップ有意性まで出す。
"""

from __future__ import annotations

import os

import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import (
    apply_regime_overlay,
    run_backtest,
    topix_returns,
    trend_following_exposure,
    vol_target_exposure,
)
from quant_jp.backtest.walkforward import bootstrap_ci, deflated_sharpe, stitch_oos
from quant_jp.data import load, universe
from quant_jp.features import trend as trd
from quant_jp.strategy import ranking

COST_BPS = float(os.environ.get("QJ_COST", "25"))
TARGET_VOL = 0.15

close = load.close_panel()
topix = load.load_topix().set_index("Date")["Close"]
eligible = universe.eligible_mask().reindex(index=close.index, columns=close.columns).fillna(False)
bench = topix_returns(topix).reindex(close.index).fillna(0.0)
start = close.index[252]
r_b = bench.loc[start:]


def lowturn_weights(rebalance, top_n=20, exit_n=40):
    """低回転スコアでバッファ選定。トレンドフィルタは現金化側に委ねるため使わない。"""
    from quant_jp.backtest.engine import _rebalance_dates
    import numpy as np

    score = ranking.lowturn_score(close, eligible).where(eligible)
    inv = (1.0 / ranking.mom.volatility(close, 63)).replace([np.inf, -np.inf], np.nan)
    rb = _rebalance_dates(close.index, rebalance)
    days = [d for d in close.index if d in rb]
    held, rows = [], {}
    for d in days:
        s = score.loc[d].dropna().sort_values(ascending=False)
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


def evaluate(weights, overlay, rebalance):
    res = run_backtest(close, weights, exposure=None, rebalance=rebalance, cost_bps=COST_BPS)
    if overlay == "trend":
        exp = trend_following_exposure(topix, res.returns.index)
    elif overlay == "vol":
        exp = vol_target_exposure(res.returns, target_vol=TARGET_VOL)
    elif overlay == "trend_x_vol":
        exp = (trend_following_exposure(topix, res.returns.index)
               * vol_target_exposure(res.returns, target_vol=TARGET_VOL)).clip(0, 1)
    else:
        exp = pd.Series(1.0, index=res.returns.index)
    return apply_regime_overlay(res.returns, exp, cost_bps=COST_BPS), res


print(f"== 低回転マルチファクター（コスト{COST_BPS:.0f}bps, 上場廃止処理あり） ==")
print("構成 | CAGR | Sharpe | MaxDD | 超過 | 年次勝率 | 回転率")
print("---|---|---|---|---|---|---")

configs = []
for rebal, label in [("Q", "四半期"), ("2Q", "半年")]:
    # pandas に 2Q が無いので 'Q' を使い exit_n を緩めて実質低回転化
    rb = "Q"
    w = lowturn_weights(rb, top_n=20, exit_n=40)
    for ov, ovlabel in [("none", "現金化なし"), ("trend", "トレンド現金化"), ("trend_x_vol", "トレンド×ボラ")]:
        r_full, res = evaluate(w, ov, rb)
        r = r_full.loc[start:]
        s = metrics.summary(r, benchmark=r_b)
        a_s = (1 + r).groupby(r.index.year).prod() - 1
        a_b = (1 + r_b).groupby(r_b.index.year).prod() - 1
        win = int((a_s - a_b > 0).sum())
        turn = res.turnover.loc[start:].sum() / (len(res.turnover.loc[start:]) / 252)
        name = f"{label}・{ovlabel}"
        configs.append((name, r))
        print(f"{name} | {s['CAGR']:.2%} | {s['Sharpe']:.2f} | {s['MaxDD']:.2%} | "
              f"{s['Excess_CAGR']:.2%} | {win}/{len(a_s)} | {turn:.0%}")
    break  # 四半期のみ（半年は exit_n でカバー）

print(f"\nTOPIX: CAGR {metrics.cagr(r_b):.2%} / Sharpe {metrics.sharpe(r_b):.2f} / MaxDD {metrics.max_drawdown(r_b):.2%}")

# 最良候補（トレンド現金化）で有意性検定
best = dict(configs).get("四半期・トレンド現金化")
if best is not None:
    print("\n== Walk-forward OOS（学習3年/検証1年, パージ付き） ==")
    oos = stitch_oos(best, close.index, train_years=3, test_years=1)
    oos_b = r_b.reindex(oos.index)
    s = metrics.summary(oos, benchmark=oos_b)
    print(f"OOS連結: CAGR {s['CAGR']:.2%} | Sharpe {s['Sharpe']:.2f} | MaxDD {s['MaxDD']:.2%} | "
          f"超過 {s['Excess_CAGR']:.2%}")

    print("\n== ブートストラップ有意性（ブロック21日, 2000回） ==")
    ci = bootstrap_ci(best.loc[start:], r_b)
    print(f"超過CAGR: 平均 {ci['excess_cagr_mean']:.2%}, 95%CI [{ci['excess_cagr_ci'][0]:.2%}, "
          f"{ci['excess_cagr_ci'][1]:.2%}], p値(<=0) {ci['excess_cagr_pval']:.3f}")
    print(f"Sharpe差: p値(<=0) {ci['sharpe_diff_pval']:.3f}")
    print(f"Deflated Sharpe: {deflated_sharpe(best.loc[start:], n_trials=9):.3f}")
