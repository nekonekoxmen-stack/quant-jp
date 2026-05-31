"""クロスセクション・ランキングと銘柄選定。

複数モメンタムの横断ランクを平均して合成スコアを作り、長期トレンド上にある
銘柄に限定して上位 N を等ウェイトで選ぶ。出力は日次の目標ウェイト（Date×Code,
行和=1、未投資分は含めない）。レジーム比率と現金は backtest 側で適用する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_jp.data import load, universe
from quant_jp.features import fundamentals as fnd
from quant_jp.features import momentum as mom
from quant_jp.features import trend as trd

# 既定のファクター重み（合成スコア）。日本株はバリュー/クオリティが強い。
DEFAULT_WEIGHTS = {"momentum": 0.34, "value": 0.33, "quality": 0.33}

# 低回転マルチファクター（モメンタム除外）。バリュー/クオリティ/低ボラは年1-2回更新で
# 十分なため取引コスト耐性が高い（Gemini 学術調査・reports/research 参照）。
LOWTURN_WEIGHTS = {"value": 0.40, "quality": 0.35, "lowvol": 0.25}


def value_tilt_score(
    close: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    *,
    quality_w: float = 0.0,
) -> pd.DataFrame:
    """バリュー主軸スコア（必要なら quality_w でクオリティを軽く加味）。

    検証の結果、日本株（2017-2026, ネット）ではバリューが突出して有効で、クオリティ/
    低ボラの等重み混合はむしろ希釈となった。よって既定はバリュー純（quality_w=0）。
    quality_w を上げるとバリュー罠（割安だが質の低い銘柄）の足切りとして機能する。
    """
    v = fnd.value_score(close, mask).fillna(0.5)
    if quality_w <= 0:
        return v
    q = fnd.quality_score(close, mask).fillna(0.5)
    return (1 - quality_w) * v + quality_w * q


def lowturn_score(
    close: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """低回転マルチファクター合成（バリュー＋クオリティ＋低ボラの統合スコア）。

    各ファクターは [0,1] パーセンタイル順位。欠損は中立0.5で補完し暗黙の選別を防ぐ。
    モメンタムを含まないため銘柄入れ替えが緩慢＝低回転でコスト控除後の優位を狙う。
    """
    w = weights or LOWTURN_WEIGHTS
    value = fnd.value_score(close, mask).fillna(0.5)
    quality = fnd.quality_score(close, mask).fillna(0.5)
    lowvol = fnd.low_volatility_score(close, mask).fillna(0.5)
    return w["value"] * value + w["quality"] * quality + w["lowvol"] * lowvol


def _rank(panel: pd.DataFrame, mask, sectors) -> pd.DataFrame:
    """セクター指定があればセクター内、無ければ全体の横断ランク。"""
    if sectors is not None:
        return mom.sector_neutral_rank(panel, sectors, mask)
    return mom.cross_sectional_rank(panel, mask)


def momentum_score(
    close: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    *,
    residual: bool = False,
    market_close: pd.Series | None = None,
    sectors: pd.Series | None = None,
) -> pd.DataFrame:
    """モメンタムの横断（または業種内）パーセンタイル順位の平均。

    日本株は月次の短期反転（reversal）が強いため、直近1ヶ月を除外（J&T 1993）。
    residual=True なら市場βを除いた残差モメンタム（Blitz+2011）を併用してβ依存を低減。
    """
    signals = [mom.momentum_12_1(close), mom.momentum_6_1(close)]
    if residual and market_close is not None:
        signals.append(mom.residual_momentum(close, market_close))
    ranks = [_rank(s, mask, sectors) for s in signals]
    return sum(ranks) / len(ranks)


def composite_score(
    close: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    *,
    weights: dict[str, float] | None = None,
    residual_mom: bool = False,
    sector_neutral: bool = False,
    market_close: pd.Series | None = None,
) -> pd.DataFrame:
    """モメンタム＋バリュー＋クオリティの横断ランクを重み付き合成。

    各ファクターは [0,1] のパーセンタイル順位。マルチファクター化により単一
    ファクターのドローダウン（特にモメンタム・クラッシュ）を緩和し、Sharpe を底上げ。
    residual_mom=市場β除去, sector_neutral=33業種内ランクで偏り排除。
    """
    w = weights or DEFAULT_WEIGHTS
    sectors = universe.sector_map(close.columns) if sector_neutral else None
    if residual_mom and market_close is None:
        topix = load.load_topix()
        if not topix.empty:
            market_close = topix.set_index("Date")["Close"]

    mom_s = momentum_score(
        close, mask, residual=residual_mom, market_close=market_close, sectors=sectors
    )
    # バリュー/クオリティが欠損（予想EPS未提供等）の銘柄を暗黙に除外しないよう中立(0.5)で補完。
    value = _rank(_value_raw(close, mask), mask, sectors).fillna(0.5) if sector_neutral else fnd.value_score(close, mask).fillna(0.5)
    quality = _rank(_quality_raw(close, mask), mask, sectors).fillna(0.5) if sector_neutral else fnd.quality_score(close, mask).fillna(0.5)
    return w["momentum"] * mom_s + w["value"] * value + w["quality"] * quality


def _value_raw(close, mask):
    """セクター中立化前のバリュー素点（E/P と B/P の平均）。"""
    p = fnd.fundamental_panels(close)
    return (p["EP"].rank(axis=1, pct=True) + p["BP"].rank(axis=1, pct=True)) / 2


def _quality_raw(close, mask):
    """セクター中立化前のクオリティ素点（OP/TA・ROE・自己資本比率 の平均）。"""
    p = fnd.fundamental_panels(close)
    return (
        p["OP_TA"].rank(axis=1, pct=True)
        + p["ROE"].rank(axis=1, pct=True)
        + p["EqAR"].rank(axis=1, pct=True)
    ) / 3


def select_weights(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    top_n: int = 15,
    trend_window: int = 200,
    weighting: str = "inverse_vol",
    vol_window: int = 63,
    factor_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """日次の目標ウェイト。トレンド上 ∧ eligible の上位 N を採用。

    weighting='inverse_vol' で逆ボラ加重（高ボラ銘柄への集中を抑え、モメンタム・
    クラッシュの影響を緩和）。'equal' で等ウェイト。
    """
    score = composite_score(close, eligible, weights=factor_weights)

    # トレンドフィルタ: 長期線より上のみ投資対象
    trend_ok = trd.above_sma(close, trend_window)
    investable = eligible & trend_ok
    score = score.where(investable)

    # 各日で上位 N を採用
    ranks = score.rank(axis=1, ascending=False, method="first")
    picked = (ranks <= top_n).astype(float)

    if weighting == "inverse_vol":
        vol = mom.volatility(close, vol_window)
        inv = (1.0 / vol).replace([float("inf"), -float("inf")], np.nan)
        raw = picked * inv
    else:  # equal
        raw = picked

    totals = raw.sum(axis=1)
    weights = raw.div(totals.replace(0, np.nan), axis=0).fillna(0.0)
    return weights


def select_weights_buffered(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    top_n: int = 15,
    exit_n: int = 30,
    rebalance: str = "ME",
    trend_window: int = 200,
    weighting: str = "inverse_vol",
    vol_window: int = 63,
    factor_weights: dict[str, float] | None = None,
    residual_mom: bool = False,
    sector_neutral: bool = False,
    score: pd.DataFrame | None = None,
    use_trend_filter: bool = True,
) -> pd.DataFrame:
    """バッファルール付きの目標ウェイト（回転率抑制）。

    リバランス日ごとに、既存保有は合成ランクが exit_n 位以下に落ちるまで継続保有し、
    空いた枠を上位の新規銘柄で埋めて top_n 銘柄を維持。微小なランク変動による
    毎月の無駄な入れ替えを防ぎ、取引コストを大幅に削減する（研究知見の定石）。

    score を渡せばその合成スコアを使う（例: バリュー主軸）。use_trend_filter=False
    で個別銘柄トレンドフィルタを外す（現金化はブックレベルのオーバーレイに委ねる）。
    """
    from quant_jp.backtest.engine import _rebalance_dates

    if score is None:
        score = composite_score(
            close, eligible, weights=factor_weights,
            residual_mom=residual_mom, sector_neutral=sector_neutral,
        )
    investable = eligible & trd.above_sma(close, trend_window) if use_trend_filter else eligible
    score = score.where(investable)
    inv = (1.0 / mom.volatility(close, vol_window)).replace([np.inf, -np.inf], np.nan)

    rb = _rebalance_dates(close.index, rebalance)
    rebal_days = [d for d in close.index if d in rb]
    held: list[str] = []
    rows: dict[pd.Timestamp, pd.Series] = {}
    for d in rebal_days:
        s = score.loc[d].dropna().sort_values(ascending=False)
        pos = {c: i + 1 for i, c in enumerate(s.index)}
        new_held = [c for c in held if pos.get(c, 10**9) <= exit_n]  # 生存（exit_n以内）
        for c in s.index:  # 空き枠を上位新規で補充
            if len(new_held) >= top_n:
                break
            if c not in new_held:
                new_held.append(c)
        new_held = new_held[:top_n]
        held = new_held
        if weighting == "inverse_vol":
            iv = inv.loc[d, new_held].replace([np.inf, -np.inf], np.nan).dropna()
            # ボラ欠損銘柄は等加重フォールバック（枠を空費しない）
            missing = [c for c in new_held if c not in iv.index]
            if missing and len(iv) > 0:
                iv = pd.concat([iv, pd.Series(iv.median(), index=missing)])
            elif missing:
                iv = pd.Series(1.0, index=new_held)
        else:
            iv = pd.Series(1.0, index=new_held)
        w = pd.Series(0.0, index=close.columns)
        if iv.sum() > 0:
            wv = iv / iv.sum()
            # 1銘柄上限（超低ボラ銘柄への過度な集中を防ぐ）。超過分は他へ再配分。
            cap = max(2.0 / max(len(new_held), 1), 0.10)
            for _ in range(3):
                over = wv > cap
                if not over.any():
                    break
                excess = (wv[over] - cap).sum()
                wv[over] = cap
                under = ~over
                if wv[under].sum() > 0:
                    wv[under] += excess * wv[under] / wv[under].sum()
            w.loc[wv.index] = wv
        rows[d] = w
    weights = pd.DataFrame(rows).T if rows else pd.DataFrame(columns=close.columns)
    return weights.reindex(index=close.index, columns=close.columns).ffill().fillna(0.0)


def select_weights_staggered(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    top_n: int = 20,
    exit_n: int = 40,
    n_tranches: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """スタッガード・リバランス: 資金を n_tranches 分割し各トランチを月ずらしで運用。

    各トランチは四半期保有だが開始月が1ヶ月ずつズレるため、ポート全体では毎月
    1/n_tranches ずつ入れ替わる。四半期末の特定日依存・スリッページ集中を排除し
    リターンを平滑化する（Jegadeesh-Titman のオーバーラップ手法）。
    全トランチの目標ウェイトを単純平均して合成ウェイトを返す。
    """
    from quant_jp.backtest.engine import _rebalance_dates

    # 月末リバランス日を取得し、トランチごとに n_tranches 周期で割り当て
    month_ends = sorted(d for d in close.index if d in _rebalance_dates(close.index, "ME"))
    acc = None
    for t in range(n_tranches):
        tranche_days = set(month_ends[t::n_tranches])
        w = _select_on_days(
            close, eligible, tranche_days, top_n=top_n, exit_n=exit_n, **kwargs
        )
        acc = w if acc is None else acc + w
    return (acc / n_tranches).fillna(0.0)


def _select_on_days(close, eligible, rebal_day_set, *, top_n, exit_n,
                    trend_window=200, vol_window=63, score=None, use_trend_filter=False,
                    factor_weights=None):
    """指定したリバランス日集合でバッファ選定（select_weights_buffered の日付指定版）。"""
    if score is None:
        score = composite_score(close, eligible, weights=factor_weights)
    investable = eligible & trd.above_sma(close, trend_window) if use_trend_filter else eligible
    score = score.where(investable)
    inv = (1.0 / mom.volatility(close, vol_window)).replace([np.inf, -np.inf], np.nan)
    rebal_days = [d for d in close.index if d in rebal_day_set]
    held, rows = [], {}
    for d in rebal_days:
        s = score.loc[d].dropna().sort_values(ascending=False)
        pos = {c: i + 1 for i, c in enumerate(s.index)}
        new = [c for c in held if pos.get(c, 10**9) <= exit_n]
        for c in s.index:
            if len(new) >= top_n:
                break
            if c not in new:
                new.append(c)
        held = new[:top_n]
        iv = inv.loc[d, held].replace([np.inf, -np.inf], np.nan).dropna()
        missing = [c for c in held if c not in iv.index]
        if missing and len(iv) > 0:
            iv = pd.concat([iv, pd.Series(iv.median(), index=missing)])
        elif missing:
            iv = pd.Series(1.0, index=held)
        w = pd.Series(0.0, index=close.columns)
        if iv.sum() > 0:
            wv = iv / iv.sum()
            cap = max(2.0 / max(len(held), 1), 0.10)
            for _ in range(3):
                over = wv > cap
                if not over.any():
                    break
                exc = (wv[over] - cap).sum()
                wv[over] = cap
                und = ~over
                if wv[und].sum() > 0:
                    wv[und] += exc * wv[und] / wv[und].sum()
            w.loc[wv.index] = wv
        rows[d] = w
    weights = pd.DataFrame(rows).T if rows else pd.DataFrame(columns=close.columns)
    return weights.reindex(index=close.index, columns=close.columns).ffill().fillna(0.0)
