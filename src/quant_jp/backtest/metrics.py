"""パフォーマンス指標。日次リターン系列（pd.Series, index=Date）を入力とする。"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """累積資産曲線。"""
    return (1.0 + returns.fillna(0.0)).cumprod() * initial


def cagr(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    if len(eq) < 2:
        return float("nan")
    final = eq.iloc[-1]
    if final <= 0:  # 破綻（累積資産が0以下）→ 実質 -100%
        return -1.0
    years = len(eq) / TRADING_DAYS
    return final ** (1 / years) - 1.0


def annual_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std()
    return float("nan") if sd == 0 else excess.mean() / sd * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    dd = eq / eq.cummax() - 1.0
    return dd.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1.0


def calmar(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    return float("nan") if mdd == 0 else cagr(returns) / abs(mdd)


def summary(returns: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
    """主要指標をまとめて返す。benchmark があれば超過 CAGR も付ける。"""
    out = {
        "CAGR": cagr(returns),
        "AnnVol": annual_vol(returns),
        "Sharpe": sharpe(returns),
        "MaxDD": max_drawdown(returns),
        "Calmar": calmar(returns),
    }
    if benchmark is not None:
        aligned = returns.reindex(benchmark.index).fillna(0.0)
        bench = benchmark.reindex(aligned.index).fillna(0.0)
        out["Bench_CAGR"] = cagr(bench)
        out["Bench_MaxDD"] = max_drawdown(bench)
        out["Excess_CAGR"] = out["CAGR"] - out["Bench_CAGR"]
    return out


def format_summary(s: dict[str, float]) -> str:
    pct = {"CAGR", "AnnVol", "MaxDD", "Bench_CAGR", "Bench_MaxDD", "Excess_CAGR"}
    return " | ".join(
        f"{k}={v:.2%}" if k in pct else f"{k}={v:.2f}" for k, v in s.items()
    )
