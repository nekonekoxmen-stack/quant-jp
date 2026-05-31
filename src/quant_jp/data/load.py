"""データレイク（Parquet）の読み込みと正規化。

J-Quants v2 の略記カラムを扱いやすい形へ整える。価格は**調整後**（Adj*）を
既定で使用し、分割の影響を除く。下流（特徴量・バックテスト）はこのモジュール
経由でデータを取得する。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"


def _read_many(subdir: str, pattern: str) -> pd.DataFrame:
    files = sorted((DATA_DIR / subdir).glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def load_daily(adjusted: bool = True) -> pd.DataFrame:
    """全期間の日足を long 形式で返す。

    返り値カラム: Date(datetime), Code(str), Open, High, Low, Close, Volume, Turnover。
    adjusted=True で調整後(Adj*)を採用。
    """
    df = _read_many("daily_quotes", "dq_*.parquet")
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df["Code"] = df["Code"].astype(str)
    o, h, l, c, v = (
        ("AdjO", "AdjH", "AdjL", "AdjC", "AdjVo")
        if adjusted
        else ("O", "H", "L", "C", "Vo")
    )
    out = df[["Date", "Code", o, h, l, c, v, "Va"]].rename(
        columns={o: "Open", h: "High", l: "Low", c: "Close", v: "Volume", "Va": "Turnover"}
    )
    return out.sort_values(["Code", "Date"]).reset_index(drop=True)


def load_topix() -> pd.DataFrame:
    """TOPIX 指数（Date, Open, High, Low, Close）。"""
    path = DATA_DIR / "topix.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path).rename(columns={"O": "Open", "H": "High", "L": "Low", "C": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_listed() -> pd.DataFrame:
    """上場銘柄一覧（最新スナップショット）。Code は str。"""
    path = DATA_DIR / "listed_info.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["Code"] = df["Code"].astype(str)
    return df


def load_listed_history() -> pd.DataFrame:
    """月次の上場スナップショット履歴（SnapDate, Code, Mkt, ...）。"""
    path = DATA_DIR / "listed_history.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["Code"] = df["Code"].astype(str)
    df["SnapDate"] = pd.to_datetime(df["SnapDate"])
    return df


def load_statements() -> pd.DataFrame:
    """財務サマリ（全期間）。DiscDate は datetime。"""
    df = _read_many("statements", "st_*.parquet")
    if df.empty:
        return df
    if "DiscDate" in df.columns:
        df["DiscDate"] = pd.to_datetime(df["DiscDate"])
    if "Code" in df.columns:
        df["Code"] = df["Code"].astype(str)
    return df


def load_investor_types() -> pd.DataFrame:
    """投資部門別売買。"""
    path = DATA_DIR / "investor_types.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@lru_cache(maxsize=4)
def close_panel(adjusted: bool = True) -> pd.DataFrame:
    """調整後終値の wide パネル（index=Date, columns=Code）。"""
    daily = load_daily(adjusted=adjusted)
    return daily.pivot(index="Date", columns="Code", values="Close").sort_index()


@lru_cache(maxsize=4)
def turnover_panel() -> pd.DataFrame:
    """売買代金の wide パネル（流動性フィルタ用）。index=Date, columns=Code。"""
    daily = load_daily()
    return daily.pivot(index="Date", columns="Code", values="Turnover").sort_index()


@lru_cache(maxsize=4)
def raw_close_panel() -> pd.DataFrame:
    """生（無調整）終値の wide パネル。値がさ株（単元コスト）判定など、当時の実株価が
    必要な用途に使う。調整後終値は分割で過去が下方修正されるため不可。"""
    daily = load_daily(adjusted=False)
    return daily.pivot(index="Date", columns="Code", values="Close").sort_index()
