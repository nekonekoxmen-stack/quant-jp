"""売買可能ユニバースの定義。

東証プライム（Mkt=='0111'）に限定し、ETF/REIT 等（その他）を除外。さらに
トレーリングの売買代金（流動性）でフィルタし、300万円規模で無理なく約定でき
薄商い銘柄を排除する。出力は close パネルに整合する真偽マスク（Date×Code）。
"""

from __future__ import annotations

import pandas as pd

from quant_jp.data import load

# 主力市場（大型主板）の市場コード: 再編前=東証一部(0101)、再編後=プライム(0111)
MAIN_BOARD_MKT = {"0101", "0111"}
DEFAULT_MIN_ADV_YEN = 1e8  # 平均売買代金 1億円/日 以上
DEFAULT_LIQ_WINDOW = 60  # 営業日
LOT_SIZE = 100  # 単元株
DEFAULT_MAX_LOT_COST_YEN = 1e6  # 1単元(100株)が100万円超の値がさ株を除外


def sector_map(columns) -> pd.Series:
    """各銘柄コード→33業種コード(S33)の対応（最新スナップショット）。

    セクター中立化に使用。履歴で業種は概ね不変のため最新値で十分。
    """
    listed = load.load_listed()
    if listed.empty or "S33" not in listed.columns:
        hist = load.load_listed_history()
        if hist.empty or "S33" not in hist.columns:
            return pd.Series(index=columns, dtype=object)
        listed = hist.sort_values("SnapDate").groupby("Code").last().reset_index()
    s = listed.set_index("Code")["S33"].astype(str)
    return s.reindex(columns)


def membership_mask(index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """point-in-time の主力市場メンバーシップ（Date×Code 真偽, ETF除外）。

    月次スナップショット履歴を前方補完して各営業日の構成を再現し、
    サバイバーシップ・バイアス（現在の構成で過去を評価する誤り）を排除する。
    """
    hist = load.load_listed_history()
    if hist.empty:
        return pd.DataFrame(False, index=index, columns=columns)
    main = hist[hist["Mkt"].isin(MAIN_BOARD_MKT)].copy()
    if "S33Nm" in main.columns:
        main = main[main["S33Nm"] != "その他"]
    main["flag"] = True
    wide = main.pivot_table(
        index="SnapDate", columns="Code", values="flag", aggfunc="last", fill_value=False
    ).astype(bool)
    wide = wide.sort_index().reindex(index.union(wide.index)).ffill().reindex(index)
    return wide.reindex(columns=columns).fillna(False).astype(bool)


def liquidity_mask(
    turnover: pd.DataFrame | None = None,
    *,
    window: int = DEFAULT_LIQ_WINDOW,
    min_adv_yen: float = DEFAULT_MIN_ADV_YEN,
) -> pd.DataFrame:
    """トレーリング中央値売買代金が閾値以上か（Date×Code の真偽パネル）。

    中央値は一時的な出来高急増に頑健。1日ずらして当日のルックアヘッドを避ける。
    """
    turnover = load.turnover_panel() if turnover is None else turnover
    med = turnover.rolling(window, min_periods=max(5, window // 2)).median()
    return (med.shift(1) >= min_adv_yen).fillna(False)


def lot_affordable_mask(
    close: pd.DataFrame | None = None,
    *,
    max_lot_cost_yen: float = DEFAULT_MAX_LOT_COST_YEN,
) -> pd.DataFrame:
    """1単元(100株)の購入額が上限以下か（値がさ株除外, Date×Code の真偽）。

    300万円・100株単位で現実的に分散保有できない超高価格株を除く。単元未満株
    （S株/かぶミニ）を使う前提なら閾値は緩めてよい。前日終値で判定。
    """
    close = load.raw_close_panel() if close is None else close
    lot_cost = close * LOT_SIZE
    return (lot_cost.shift(1) <= max_lot_cost_yen).fillna(False)


def eligible_mask(
    *,
    window: int = DEFAULT_LIQ_WINDOW,
    min_adv_yen: float = DEFAULT_MIN_ADV_YEN,
    max_lot_cost_yen: float = DEFAULT_MAX_LOT_COST_YEN,
) -> pd.DataFrame:
    """主力市場(PIT) ∧ 流動性 ∧ 単元購入可能 を満たす銘柄の真偽パネル（Date×Code）。"""
    liq = liquidity_mask(window=window, min_adv_yen=min_adv_yen)
    member = membership_mask(liq.index, liq.columns)
    affordable = lot_affordable_mask(max_lot_cost_yen=max_lot_cost_yen).reindex(
        index=liq.index, columns=liq.columns
    ).fillna(False)
    return liq & member & affordable


def summary() -> pd.DataFrame:
    """日次の対象銘柄数を返す（健全性チェック用）。"""
    mask = eligible_mask()
    return mask.sum(axis=1).rename("n_eligible").to_frame()
