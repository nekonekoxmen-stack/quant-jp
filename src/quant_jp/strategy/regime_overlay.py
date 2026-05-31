"""相場レジーム検知と株式エクスポージャーのダイヤル（0..1）。

弱気相場の予兆で株式比率を絞り現金化する。複数シグナルを合成:
  1. 指数トレンド: TOPIX が長期移動平均より上か＋傾き
  2. ブレッドス: ユニバースのうち長期線より上の銘柄比率
  3. ボラ・レジーム: 実現ボラが高いほどエクスポージャー縮小（ボラ・ターゲティング）
  4. ドローダウン・スロットル: 指数の高値からの下落が深いほど縮小
最終的に最大DD≈-20%以内へ収まるよう係数を調整する。初期はルールベース。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _drawdown(close: pd.Series) -> pd.Series:
    return close / close.cummax() - 1.0


def realized_vol(close: pd.Series, window: int = 21) -> pd.Series:
    logret = np.log(close / close.shift(1))
    return logret.rolling(window, min_periods=window // 2).std() * np.sqrt(252)


def market_exposure(
    topix_close: pd.Series,
    breadth: pd.Series | None = None,
    *,
    ma_window: int = 200,
    target_vol: float = 0.15,
    max_vol_leverage: float = 1.0,
    breadth_floor: float = 0.40,
    dd_soft: float = -0.08,
    dd_hard: float = -0.20,
    smooth_span: int = 5,
) -> pd.DataFrame:
    """日次の株式エクスポージャー(0..1)と各構成要素を DataFrame で返す。

    返り値カラム: trend, vol_factor, breadth_factor, dd_factor, raw, exposure。
    すべて shift(1) 済みの観測値から計算し、当日のルックアヘッドを避ける。
    """
    close = topix_close.sort_index()

    # 1) トレンド: 200日線より上 かつ 傾き>=0 で 1、それ以外は 0（前日までで判定）
    ma = close.rolling(ma_window, min_periods=ma_window // 2).mean()
    slope = ma / ma.shift(21) - 1.0
    trend = ((close > ma) & (slope >= 0)).astype(float)
    trend = trend.shift(1)

    # 2) ボラ・ターゲティング: target/realized を [0, max] にクリップ
    rv = realized_vol(close).shift(1)
    vol_factor = (target_vol / rv).clip(upper=max_vol_leverage)
    vol_factor = vol_factor.clip(lower=0.0).fillna(0.0)

    # 3) ブレッドス: floor 未満で線形に 0 へ
    if breadth is not None:
        b = breadth.reindex(close.index).shift(1)
        breadth_factor = (b / breadth_floor).clip(upper=1.0).fillna(1.0)
    else:
        breadth_factor = pd.Series(1.0, index=close.index)

    # 4) ドローダウン・スロットル: soft..hard を 1..0 へ線形
    dd = _drawdown(close).shift(1)
    dd_factor = ((dd - dd_hard) / (dd_soft - dd_hard)).clip(lower=0.0, upper=1.0).fillna(1.0)

    raw = trend * vol_factor * breadth_factor * dd_factor
    exposure = raw.ewm(span=smooth_span, adjust=False).mean().clip(0.0, 1.0)

    return pd.DataFrame(
        {
            "trend": trend,
            "vol_factor": vol_factor,
            "breadth_factor": breadth_factor,
            "dd_factor": dd_factor,
            "raw": raw,
            "exposure": exposure,
        }
    )
