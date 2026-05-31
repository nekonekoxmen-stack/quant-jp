"""モメンタム特徴量（クロスセクション）。

すべて wide パネル（index=Date, columns=Code）を受け取り、同形のパネルを返す。
リバランス時点で観測可能な値のみを使う（当日終値ベース、未来参照なし）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 営業日換算の代表的なルックバック
M1, M3, M6, M12 = 21, 63, 126, 252


def total_return(close: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """直近 lookback 営業日のトータルリターン。skip で直近を除外（反転回避）。"""
    return close.shift(skip) / close.shift(lookback) - 1.0


def momentum_12_1(close: pd.DataFrame) -> pd.DataFrame:
    """12-1 モメンタム（過去12ヶ月、直近1ヶ月除外）。最も実績のある定番。"""
    return total_return(close, lookback=M12, skip=M1)


def momentum_6_1(close: pd.DataFrame) -> pd.DataFrame:
    return total_return(close, lookback=M6, skip=M1)


def momentum_3_0(close: pd.DataFrame) -> pd.DataFrame:
    return total_return(close, lookback=M3, skip=0)


def volatility(close: pd.DataFrame, window: int = M3) -> pd.DataFrame:
    """日次対数リターンの実現ボラ（年率）。ボラ調整やサイジングに使用。"""
    logret = np.log(close / close.shift(1))
    return logret.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(252)


def residual_momentum(
    close: pd.DataFrame,
    market_close: pd.Series,
    *,
    beta_window: int = 252,
    lookback: int = M12,
    skip: int = M1,
) -> pd.DataFrame:
    """市場（TOPIX）ベータを除いた残差リターンの 12-1 モメンタム。

    各銘柄の日次対数リターンを市場リターンに回帰（ローリング beta_window）し、
    残差 = r - beta*r_mkt の累積（直近 skip を除外）。市場・セクターβ由来の
    見かけのモメンタムを排し、個別アルファを抽出（Blitz, Huij & Martens 2011）。
    実装は閉形式の共分散/分散でベータを推定（高速）。
    """
    r = np.log(close / close.shift(1))
    m = np.log(market_close / market_close.shift(1)).reindex(r.index)
    mean_m = m.rolling(beta_window, min_periods=beta_window // 2).mean()
    var_m = m.rolling(beta_window, min_periods=beta_window // 2).var()
    # cov(r_i, m) をローリングで: E[r*m]-E[r]E[m]
    rm = r.mul(m, axis=0)
    mean_rm = rm.rolling(beta_window, min_periods=beta_window // 2).mean()
    mean_r = r.rolling(beta_window, min_periods=beta_window // 2).mean()
    cov = mean_rm.sub(mean_r.mul(mean_m, axis=0))
    beta = cov.div(var_m.replace(0, np.nan), axis=0)
    resid = r.sub(beta.mul(m, axis=0))
    # 残差の 12-1 累積（skip 日除外）
    cum = resid.shift(skip).rolling(lookback - skip, min_periods=(lookback - skip) // 2).sum()
    return cum


def cross_sectional_zscore(panel: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """各日の銘柄横断 z-score。mask=True の銘柄のみで統計量を計算。"""
    valid = panel.where(mask) if mask is not None else panel
    mu = valid.mean(axis=1)
    sd = valid.std(axis=1)
    z = valid.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z


def cross_sectional_rank(panel: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """各日の銘柄横断パーセンタイル順位（0..1）。外れ値に頑健。"""
    valid = panel.where(mask) if mask is not None else panel
    return valid.rank(axis=1, pct=True)


def sector_neutral_rank(
    panel: pd.DataFrame,
    sectors: pd.Series,
    mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """セクター内のパーセンタイル順位（0..1）。

    各業種グループ内で順位付けし、特定セクターへの偏り（例: バリュー→銀行・商社集中）
    を排除。sectors は Code→業種コードの対応。グループ単位で rank(pct) を計算。
    """
    valid = panel.where(mask) if mask is not None else panel
    out = pd.DataFrame(np.nan, index=valid.index, columns=valid.columns)
    sec = sectors.reindex(valid.columns)
    for _, cols in sec.dropna().groupby(sec.dropna()):
        members = list(cols.index)
        if len(members) >= 3:  # 統計が無意味な極小グループは除外
            out[members] = valid[members].rank(axis=1, pct=True)
    return out
