"""ファンダメンタル特徴量（バリュー／クオリティ）を point-in-time で構築。

財務は開示日(DiscDate)以降にのみ利用可能。各営業日について「その時点で開示済みの
最新値」を前方補完し、さらに翌営業日から有効化して先読みを防ぐ。

高カバレッジ（非欠損率〜83%）の項目のみ採用し、欠損による暗黙の銘柄選別を避ける:
  - Value   : 益利回り E/P = EPS/Price、簿価株価倍率 B/P = (Eq/発行株数)/Price
  - Quality : 収益性 OP/TA（総資産営業利益率, Novy-Marx 流）、ROE = NP/Eq、自己資本比率 EqAR

低回転ファクター（年1〜2回更新で十分）。GP/A の粗利は J-Quants サマリに無いため
営業利益/総資産で代替。配当・CFO は欠損が多く（<45%）見送り。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_jp.data import load
from quant_jp.features import momentum as mom


def _asof_panel(stmts: pd.DataFrame, col: str, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """開示日ベースで最新値を前方補完した Date×Code パネル（翌営業日から有効）。"""
    s = stmts[["DiscDate", "Code", col]].copy()
    s[col] = pd.to_numeric(s[col], errors="coerce")
    s = s.dropna(subset=[col])
    if s.empty:
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    wide = s.pivot_table(index="DiscDate", columns="Code", values=col, aggfunc="last")
    wide = wide.sort_index().reindex(index.union(wide.index)).ffill().reindex(index)
    wide = wide.shift(1)  # 大引け後開示が大半 → 翌営業日から有効化
    return wide.reindex(columns=columns)


def _ttm_eps_table(stmts: pd.DataFrame) -> pd.DataFrame:
    """開示ごとの TTM(過去4四半期累計) EPS を返す（列: Code, DiscDate, ttm_eps）。

    J-Quants の EPS は会計年度の期初来累計（1Q→2Q→3Q→FY でリセット）。横断比較で
    「会計年度のどこにいるか」の季節性を拾わないよう、各社の四半期 EPS（=累計の差分、
    1Q はそのまま）を求め、過去4四半期分を移動合計して TTM 益を作る。
    """
    s = stmts[["DiscDate", "Code", "CurPerType", "EPS"]].copy()
    s["EPS"] = pd.to_numeric(s["EPS"], errors="coerce")
    s = s.dropna(subset=["EPS"]).sort_values(["Code", "DiscDate"])
    # 同一(Code,DiscDate)重複は最後を採用
    s = s.drop_duplicates(["Code", "DiscDate"], keep="last")

    order = {"1Q": 1, "2Q": 2, "3Q": 3, "4Q": 4, "5Q": 4, "FY": 4}
    s["q"] = s["CurPerType"].map(order)

    def _per_code(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        # 四半期単独 EPS = 累計の差分。ただし 1Q(期初) は累計=単独。
        prev = g["EPS"].shift(1)
        is_q1 = g["q"] == 1
        q_eps = g["EPS"] - prev.where(~is_q1, 0.0)
        # 期跨ぎ（前行が同年度でない=1Q）の差分は EPS そのもの
        q_eps = q_eps.where(~is_q1, g["EPS"])
        g["q_eps"] = q_eps
        g["ttm_eps"] = g["q_eps"].rolling(4, min_periods=4).sum()
        return g

    out = s.groupby("Code", group_keys=False).apply(_per_code)
    return out[["Code", "DiscDate", "ttm_eps"]].dropna(subset=["ttm_eps"])


def _ttm_eps_panel(stmts: pd.DataFrame, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """TTM EPS の as-of パネル（翌営業日から有効）。"""
    tbl = _ttm_eps_table(stmts)
    if tbl.empty:
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    wide = tbl.pivot_table(index="DiscDate", columns="Code", values="ttm_eps", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().reindex(index.union(wide.index)).ffill().reindex(index)
    wide = wide.shift(1)
    return wide.reindex(columns=columns)


def fundamental_panels(close: pd.DataFrame, stmts: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """高カバレッジの PIT ファンダ・パネルを返す。

    返り値: EP(益利回り), BP(簿価株価倍率), OP_TA(総資産営業利益率), ROE, EqAR。
    """
    stmts = load.load_statements() if stmts is None else stmts
    idx, cols = close.index, close.columns
    ttm_eps = _ttm_eps_panel(stmts, idx, cols)        # TTM EPS（季節性除去）
    eq = _asof_panel(stmts, "Eq", idx, cols)          # 自己資本（83%）
    shout = _asof_panel(stmts, "ShOutFY", idx, cols)  # 期末発行株数（83%）
    trsh = _asof_panel(stmts, "TrShFY", idx, cols)    # 自己株式（79%）
    op = _asof_panel(stmts, "OP", idx, cols)          # 営業利益（81%, YTD累計）
    np_ = _asof_panel(stmts, "NP", idx, cols)         # 純利益（83%, YTD累計）
    ta = _asof_panel(stmts, "TA", idx, cols)          # 総資産（83%）
    eqar = _asof_panel(stmts, "EqAR", idx, cols)      # 自己資本比率（83%, ストック）

    shares = (shout - trsh.fillna(0)).replace(0, np.nan)  # 自己株控除後の発行株数
    bvps = eq / shares                                    # 一株当たり純資産（ストック, 季節性なし）

    ep = ttm_eps / close                              # TTM 益利回り
    bp = bvps / close
    # 収益性・ROE は YTD 累計のままだと水準が会計年度内で増加するが、横断ランクでの
    # 相対比較かつ EqAR(ストック)と併せて使うため、ここでは ROE は補助的に扱う。
    op_ta = op / ta.replace(0, np.nan)
    roe = np_ / eq.replace(0, np.nan)
    return {"EP": ep, "BP": bp, "OP_TA": op_ta, "ROE": roe, "EqAR": eqar}


def value_score(close: pd.DataFrame, mask: pd.DataFrame | None = None, stmts=None) -> pd.DataFrame:
    """バリュー合成（益利回り E/P と 簿価株価倍率 B/P の横断ランク平均）。"""
    p = fundamental_panels(close, stmts)
    ranks = [mom.cross_sectional_rank(p["EP"], mask), mom.cross_sectional_rank(p["BP"], mask)]
    return sum(ranks) / len(ranks)


def quality_score(close: pd.DataFrame, mask: pd.DataFrame | None = None, stmts=None) -> pd.DataFrame:
    """クオリティ合成（総資産営業利益率・ROE・自己資本比率 の横断ランク平均）。"""
    p = fundamental_panels(close, stmts)
    ranks = [
        mom.cross_sectional_rank(p["OP_TA"], mask),
        mom.cross_sectional_rank(p["ROE"], mask),
        mom.cross_sectional_rank(p["EqAR"], mask),
    ]
    return sum(ranks) / len(ranks)


def low_volatility_score(
    close: pd.DataFrame, mask: pd.DataFrame | None = None, window: int = 252
) -> pd.DataFrame:
    """低ボラティリティ合成（実現ボラの低さ＝高ランク）。日本株で頑健なアノマリー。"""
    vol = mom.volatility(close, window)
    return mom.cross_sectional_rank(-vol, mask)
