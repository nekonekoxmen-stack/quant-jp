"""Purged / Embargoed walk-forward 検証とブートストラップ有意性。

金融時系列の情報リークを避けるため、学習区間と検証区間の間に「特徴量の
ルックバック長ぶんのパージ」と「エンバーゴ」を挟む（López de Prado 2018）。
本戦略はパラメータ学習を伴わないルールベースだが、設計選択（重み・ボラ目標）の
頑健性を OOS で逐次確認する枠組みとして用いる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_jp.backtest import metrics

EMBARGO_DAYS = 252  # モメンタム/移動平均の最長ルックバック相当


def walk_forward_slices(
    index: pd.DatetimeIndex,
    *,
    train_years: int = 3,
    test_years: int = 1,
    embargo: int = EMBARGO_DAYS,
):
    """(train_idx, test_idx) のリストを返す（拡張窓・パージ付き）。"""
    start = index.min()
    slices = []
    test_start = start + pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
    while test_start < index.max():
        test_end = test_start + pd.DateOffset(years=test_years)
        train_mask = index < (test_start - pd.Timedelta(days=int(embargo * 365 / 252)))
        test_mask = (index >= test_start) & (index < test_end)
        if test_mask.sum() > 20 and train_mask.sum() > 250:
            slices.append((index[train_mask], index[test_mask]))
        test_start = test_end
    return slices


def stitch_oos(returns: pd.Series, index: pd.DatetimeIndex, **kw) -> pd.Series:
    """walk-forward の各テスト区間を連結した OOS リターン系列。"""
    slices = walk_forward_slices(index, **kw)
    parts = [returns.reindex(test) for _, test in slices]
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def bootstrap_ci(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    n: int = 2000,
    block: int = 21,
    seed: int = 42,
) -> dict:
    """ブロック・ブートストラップで超過CAGR・Sharpeの95%信頼区間とp値。

    時系列の自己相関を保つため block 日単位でリサンプル。p値は「超過CAGR<=0 /
    Sharpe差<=0 となる割合」（片側）。
    """
    rng = np.random.default_rng(seed)
    r = returns.fillna(0.0).to_numpy()
    b = benchmark.reindex(returns.index).fillna(0.0).to_numpy()
    m = len(r)
    n_blocks = int(np.ceil(m / block))
    exc_cagr, sharpe_diff = [], []
    for _ in range(n):
        starts = rng.integers(0, m - block, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:m]
        rs, bs = pd.Series(r[idx]), pd.Series(b[idx])
        exc_cagr.append(metrics.cagr(rs) - metrics.cagr(bs))
        sharpe_diff.append(metrics.sharpe(rs) - metrics.sharpe(bs))
    exc = np.array(exc_cagr)
    shd = np.array(sharpe_diff)
    return {
        "excess_cagr_mean": float(exc.mean()),
        "excess_cagr_ci": (float(np.percentile(exc, 2.5)), float(np.percentile(exc, 97.5))),
        "excess_cagr_pval": float((exc <= 0).mean()),
        "sharpe_diff_mean": float(shd.mean()),
        "sharpe_diff_ci": (float(np.percentile(shd, 2.5)), float(np.percentile(shd, 97.5))),
        "sharpe_diff_pval": float((shd <= 0).mean()),
    }


def deflated_sharpe(returns: pd.Series, n_trials: int) -> float:
    """Deflated Sharpe Ratio（複数検定で水増しされた Sharpe を割引）の近似 p 値。

    観測 Sharpe が「n_trials 回試行のうちのベスト」だった場合に、真の Sharpe>0 で
    ある確率の近似（López de Prado 2014 を簡略化）。
    """
    from scipy.stats import norm

    sr = metrics.sharpe(returns) / np.sqrt(252)  # 日次 Sharpe
    t = len(returns)
    if t < 30:
        return float("nan")
    # 複数検定で期待される最大 Sharpe（標準正規の順序統計量近似）
    e_max = (1 - np.euler_gamma) * norm.ppf(1 - 1.0 / n_trials) + np.euler_gamma * norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    sr_std = np.sqrt((1 + 0.5 * sr**2) / (t - 1))
    dsr = norm.cdf((sr - e_max * sr_std) / sr_std)
    return float(dsr)
