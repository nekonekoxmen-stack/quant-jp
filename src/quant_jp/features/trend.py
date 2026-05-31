"""トレンド特徴量（個別銘柄・指数）。

下落銘柄の除外フィルタや、相場レジーム（指数トレンド・ブレッドス）の素材。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """単純移動平均。"""
    return close.rolling(window, min_periods=max(5, window // 2)).mean()


def above_sma(close: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """終値が移動平均より上か（真偽パネル, bool）。トレンドフィルタの定番。

    移動平均が未確定/価格欠損の箇所は False（投資対象外）。
    """
    ma = sma(close, window)
    return (close > ma) & ma.notna() & close.notna()


def sma_slope(close: pd.DataFrame, window: int = 200, span: int = 21) -> pd.DataFrame:
    """移動平均の傾き（span 営業日前比）。プラスで上昇トレンド。"""
    ma = sma(close, window)
    return ma / ma.shift(span) - 1.0


def breadth_above_sma(
    close: pd.DataFrame, window: int = 200, mask: pd.DataFrame | None = None
) -> pd.Series:
    """ユニバースのうち移動平均より上にある銘柄比率（レジーム用ブレッドス）。"""
    ma = sma(close, window)
    valid = ma.notna() & close.notna()
    above = (close > ma) & valid
    if mask is not None:
        valid = valid & mask
        above = above & mask
    denom = valid.sum(axis=1).replace(0, np.nan)
    return (above.sum(axis=1) / denom).rename(f"breadth_above_sma{window}")
