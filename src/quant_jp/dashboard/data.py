"""ダッシュボード用のデータ生成（Streamlit 非依存の純関数）。

戦略の計算（ユニバース→シグナル→バックテスト→現金化→指標→当日PF）を一括で行い、
表示に必要な要素を dict で返す。重い処理はここに集約し、UI 側はキャッシュして呼ぶ。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import (
    apply_overlay_full,
    book_trend_exposure,
    run_backtest,
    topix_returns,
)
from quant_jp.data import load, universe
from quant_jp.strategy import ranking

LOT = 100


@dataclass
class DashboardData:
    asof: pd.Timestamp
    exposure_now: float
    portfolio: pd.DataFrame
    equity: pd.DataFrame  # columns: strategy, full, topix
    exposure_series: pd.Series
    annual: pd.DataFrame  # index year, columns strategy/topix/excess
    summary_strategy: dict
    summary_full: dict
    summary_topix: dict
    invested_yen: float
    cash_yen: float


def build(
    capital: float = 3_000_000,
    top_n: int = 20,
    quality_w: float = 0.0,
    cost_bps: float = 25.0,
    rebalance: str = "Q",
) -> DashboardData:
    close = load.close_panel()
    topix = load.load_topix().set_index("Date")["Close"]
    listed = load.load_listed().set_index("Code")
    eligible = universe.eligible_mask().reindex(
        index=close.index, columns=close.columns
    ).fillna(False)

    score = ranking.value_tilt_score(close, eligible, quality_w=quality_w)
    weights = ranking.select_weights_staggered(
        close, eligible, top_n=top_n, exit_n=2 * top_n,
        score=score, use_trend_filter=False,
    )
    res_inv = run_backtest(close, weights, exposure=None, rebalance="ME", cost_bps=cost_bps)
    exposure = book_trend_exposure(res_inv.gross_returns, ma_window=200, low=0.30)
    r_regime = apply_overlay_full(res_inv, exposure, cost_bps=cost_bps, cash_annual_rate=0.005)
    full_exp = pd.Series(1.0, index=res_inv.gross_returns.index)
    r_inv = apply_overlay_full(res_inv, full_exp, cost_bps=cost_bps, cash_annual_rate=0.005)
    bench = topix_returns(topix).reindex(close.index).fillna(0.0)

    start = close.index[252]
    r_s, r_f, r_b = r_regime.loc[start:], r_inv.loc[start:], bench.loc[start:]
    exp = exposure.loc[start:]

    # 当日の推奨ポートフォリオ（株価・株数は実発注用に生株価で算出）
    asof = close.index[-1]
    w = weights.loc[asof]
    w = w[w > 0].sort_values(ascending=False)
    exposure_now = float(exposure.loc[asof])
    raw_px = load.raw_close_panel()
    raw_row = raw_px.loc[asof] if asof in raw_px.index else None
    adj_row = close.loc[asof]
    rows = []
    for code, wt in w.items():
        price = raw_row.get(code, np.nan) if raw_row is not None else np.nan
        if not np.isfinite(price) or price <= 0:
            price = adj_row.get(code, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        target_yen = capital * wt * exposure_now * 0.99
        shares = int(target_yen / price)
        name = str(listed["CoName"].get(code, "")) if not listed.empty else ""
        rows.append(
            {"コード": code, "銘柄": name, "比率": wt, "株価": price,
             "株数": shares, "金額": shares * price}
        )
    portfolio = pd.DataFrame(rows)
    invested = float(portfolio["金額"].sum()) if not portfolio.empty else 0.0

    # 資産曲線
    equity = pd.DataFrame(
        {
            "strategy": metrics.equity_curve(r_s),
            "full": metrics.equity_curve(r_f),
            "topix": metrics.equity_curve(r_b),
        }
    )

    # 年次
    def annual(r: pd.Series) -> pd.Series:
        return (1 + r).groupby(r.index.year).prod() - 1

    a_s, a_b = annual(r_s), annual(r_b)
    annual_df = pd.DataFrame({"strategy": a_s, "topix": a_b})
    annual_df["excess"] = annual_df["strategy"] - annual_df["topix"]

    return DashboardData(
        asof=asof,
        exposure_now=exposure_now,
        portfolio=portfolio,
        equity=equity,
        exposure_series=exp,
        annual=annual_df,
        summary_strategy=metrics.summary(r_s, benchmark=r_b),
        summary_full=metrics.summary(r_f, benchmark=r_b),
        summary_topix=metrics.summary(r_b),
        invested_yen=invested,
        cash_yen=capital - invested,
    )
