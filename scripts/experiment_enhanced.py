"""監査残課題の全修正を統合評価。

ベースライン（修正後の素戦略）vs 強化版（残差モメンタム＋セクター中立＋V字対策の
高速ボラ目標）を、上場廃止処理・保守的コスト(25bps)込みで比較し、
walk-forward OOS とブートストラップ有意性・Deflated Sharpe まで出す。
"""

from __future__ import annotations

import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import (
    apply_regime_overlay,
    run_backtest,
    topix_returns,
    vol_target_exposure,
    vol_target_exposure_fast,
)
from quant_jp.backtest.walkforward import bootstrap_ci, deflated_sharpe, stitch_oos
from quant_jp.data import load, universe
from quant_jp.strategy import ranking

import os

COST_BPS = float(os.environ.get("QJ_COST", "25"))
REBAL, TARGET_VOL = "ME", 0.15

close = load.close_panel()
topix = load.load_topix().set_index("Date")["Close"]
eligible = universe.eligible_mask().reindex(index=close.index, columns=close.columns).fillna(False)
bench = topix_returns(topix).reindex(close.index).fillna(0.0)
start = close.index[252]
r_b = bench.loc[start:]


def strat_returns(*, residual, sector, fast):
    w = ranking.select_weights_buffered(
        close, eligible, top_n=15, exit_n=30, rebalance=REBAL,
        residual_mom=residual, sector_neutral=sector,
    )
    res = run_backtest(close, w, exposure=None, rebalance=REBAL, cost_bps=COST_BPS)
    if fast:
        exp = vol_target_exposure_fast(res.returns, topix, target_vol=TARGET_VOL)
    else:
        exp = vol_target_exposure(res.returns, target_vol=TARGET_VOL)
    return apply_regime_overlay(res.returns, exp, cost_bps=COST_BPS)


configs = {
    "ベースライン(修正後)": dict(residual=False, sector=False, fast=False),
    "+残差モメンタム": dict(residual=True, sector=False, fast=False),
    "+セクター中立": dict(residual=True, sector=True, fast=False),
    "+V字対策(高速ボラ)": dict(residual=True, sector=True, fast=True),
}

print(f"== 全修正の統合比較（コスト{COST_BPS:.0f}bps, 上場廃止処理あり） ==")
print("構成 | CAGR | Sharpe | MaxDD | 超過 | 年次勝率")
print("---|---|---|---|---|---")
results = {}
for name, cfg in configs.items():
    r = strat_returns(**cfg).loc[start:]
    results[name] = r
    s = metrics.summary(r, benchmark=r_b)
    a_s = (1 + r).groupby(r.index.year).prod() - 1
    a_b = (1 + r_b).groupby(r_b.index.year).prod() - 1
    win = int((a_s - a_b > 0).sum())
    print(f"{name} | {s['CAGR']:.2%} | {s['Sharpe']:.2f} | {s['MaxDD']:.2%} | "
          f"{s['Excess_CAGR']:.2%} | {win}/{len(a_s)}")

print(f"\n（参考）TOPIX CAGR {metrics.cagr(r_b):.2%} / Sharpe {metrics.sharpe(r_b):.2f} / MaxDD {metrics.max_drawdown(r_b):.2%}")

# 最良構成で walk-forward OOS とブートストラップ
best = results["+V字対策(高速ボラ)"]
print("\n== Walk-forward OOS（学習3年/検証1年, パージ付き） ==")
oos = stitch_oos(best, close.index, train_years=3, test_years=1)
oos_b = r_b.reindex(oos.index)
s_oos = metrics.summary(oos, benchmark=oos_b)
print(f"OOS連結: CAGR {s_oos['CAGR']:.2%} | Sharpe {s_oos['Sharpe']:.2f} | "
      f"MaxDD {s_oos['MaxDD']:.2%} | 超過 {s_oos['Excess_CAGR']:.2%}（期間 {oos.index[0].date()}〜{oos.index[-1].date()}）")

print("\n== ブートストラップ有意性（ブロック21日, 2000回） ==")
ci = bootstrap_ci(best, r_b)
print(f"超過CAGR: 平均 {ci['excess_cagr_mean']:.2%}, 95%CI [{ci['excess_cagr_ci'][0]:.2%}, "
      f"{ci['excess_cagr_ci'][1]:.2%}], p値(<=0) {ci['excess_cagr_pval']:.3f}")
print(f"Sharpe差: 平均 {ci['sharpe_diff_mean']:.2f}, 95%CI [{ci['sharpe_diff_ci'][0]:.2f}, "
      f"{ci['sharpe_diff_ci'][1]:.2f}], p値(<=0) {ci['sharpe_diff_pval']:.3f}")

dsr = deflated_sharpe(best, n_trials=len(configs) * 3)
print(f"\nDeflated Sharpe（複数検定補正後の真Sharpe>0確率）: {dsr:.3f}")
