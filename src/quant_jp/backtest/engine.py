"""クロスセクション戦略のバックテスト・エンジン。

設計上の要点（健全性）:
  - シグナルは「前日終値まで」で決定し、**翌日のリターンに適用**（ルックアヘッド排除）。
  - リバランス日のみ目標ウェイトへ更新。間の日は保有を継続しウェイトはドリフト。
  - 売買回転に対し取引コスト（手数料＋スリッページ, bps）を控除。
  - レジーム・エクスポージャー(0..1)で株式比率を縮小、残りは現金（リターン0）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series  # 日次ポートフォリオ・リターン（コスト後）
    equity: pd.Series  # 累積資産曲線
    exposure: pd.Series  # 適用した株式エクスポージャー
    turnover: pd.Series  # 日次の片道回転率
    gross_returns: pd.Series | None = None  # コスト前・フル投資のグロス日次リターン
    rebal_turnover: pd.Series | None = None  # リバランス由来の片道回転率（銘柄入替分）


def _rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> set:
    """freq ごとの最終営業日を「シグナル日」として返す（例: 'W-FRI', 'ME'）。"""
    s = pd.Series(dates, index=dates)
    grouped = s.groupby(s.dt.to_period(_period_alias(freq))).max()
    return set(grouped.values)


def _period_alias(freq: str) -> str:
    f = freq.upper()
    if f.startswith("W"):
        return "W"
    if f.startswith("M"):
        return "M"
    if f.startswith("Q"):
        return "Q"
    return "W"


def run_backtest(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    exposure: pd.Series | None = None,
    rebalance: str = "W-FRI",
    cost_bps: float = 10.0,
    delisting_return: float = -0.30,
) -> BacktestResult:
    """目標ウェイトとレジーム比率からポートフォリオ・リターンを生成する。

    上場廃止処理: 保有中に価格が消失（NaN化）した銘柄は、その日に
    `delisting_return`（既定 -30%, 上場廃止理由が不明なため保守的な定数）を
    実現してポジションを現金化する。これにより破綻・整理銘柄の損失を取りこぼさない
    （fillna(0) による上方バイアスを排除）。市場変更等の健全な退出も同率を被るが、
    多くは流動性/メンバーシップ・フィルタで事前に外れる。
    """
    close = close.sort_index()
    dates = close.index
    codes = close.columns

    close_np = close.to_numpy()
    valid = ~np.isnan(close_np)  # 価格が存在する日（上場中）
    ret = close.pct_change(fill_method=None).fillna(0.0).to_numpy()
    tw = target_weights.reindex(index=dates, columns=codes).fillna(0.0).to_numpy()
    exp = (
        exposure.reindex(dates).ffill().fillna(0.0).to_numpy()
        if exposure is not None
        else np.ones(len(dates))
    )

    rb = _rebalance_dates(dates, rebalance)
    is_signal = np.array([d in rb for d in dates])

    n = len(dates)
    w = np.zeros(len(codes))
    cost_rate = cost_bps * 1e-4
    port_rets = np.zeros(n)
    gross_rets = np.zeros(n)  # コスト前のグロス（廃止損は含む）
    turn = np.zeros(n)
    applied_exp = np.zeros(n)

    for k in range(n):
        # 上場廃止: 保有中(w>0)かつ前日まで有効・本日無効になった銘柄を現金化
        if k > 0:
            delisted = (w > 0) & valid[k - 1] & (~valid[k])
            if delisted.any():
                dl = float((w[delisted]).sum() * delisting_return)
                port_rets[k] += dl
                gross_rets[k] += dl
                w = np.where(delisted, 0.0, w)

        # 前日がシグナル日なら本日寄りでリバランス（前日終値ベースの目標）。
        # 退場済み(本日無効)の銘柄は目標から除外する。
        if k > 0 and is_signal[k - 1]:
            target = tw[k - 1] * exp[k - 1]
            target = np.where(valid[k], target, 0.0)
            t = np.abs(target - w).sum()
            turn[k] = t * 0.5  # 片道
            port_rets[k] -= t * cost_rate  # 往復コストは |Δw| 比例で近似
            w = target

        r = np.where(valid[k], np.nan_to_num(ret[k]), 0.0)
        gross = float(w @ r)
        port_rets[k] += gross
        gross_rets[k] += gross
        applied_exp[k] = w.sum()

        # ドリフト（コスト前グロスで再正規化、現金はリターン0）
        denom = 1.0 + gross
        if denom > 0:
            w = w * (1.0 + r) / denom

    rets = pd.Series(port_rets, index=dates, name="strategy")
    equity = (1.0 + rets).cumprod().rename("equity")
    return BacktestResult(
        returns=rets,
        equity=equity,
        exposure=pd.Series(applied_exp, index=dates, name="exposure"),
        turnover=pd.Series(turn, index=dates, name="turnover"),
        gross_returns=pd.Series(gross_rets, index=dates, name="gross"),
        rebal_turnover=pd.Series(turn, index=dates, name="rebal_turnover"),
    )


def topix_returns(topix_close: pd.Series) -> pd.Series:
    """TOPIX の日次リターン（ベンチマーク比較用）。"""
    return topix_close.sort_index().pct_change(fill_method=None).fillna(0.0).rename("topix")


def vol_target_exposure(
    strat_returns: pd.Series,
    *,
    target_vol: float = 0.12,
    window: int = 63,
    cap: float = 1.0,
) -> pd.Series:
    """戦略自身の実現ボラから、目標ボラに合わせる株式比率(0..cap)を返す。

    ブック固有のリスク（モメンタム・クラッシュ含む）に日次で反応する主たるリスク制御。
    レバレッジは掛けない（cap<=1）。前日までの情報で算出。
    """
    import numpy as np

    rv = strat_returns.rolling(window, min_periods=window // 2).std() * np.sqrt(252)
    scale = (target_vol / rv).clip(upper=cap)
    return scale.shift(1).clip(lower=0.0).fillna(0.0).rename("vol_target")


def vol_target_exposure_fast(
    strat_returns: pd.Series,
    market_close: pd.Series | None = None,
    *,
    target_vol: float = 0.15,
    slow_window: int = 63,
    fast_window: int = 20,
    cap: float = 1.0,
    dead_band: float = 0.10,
    reentry_ma: int = 25,
) -> pd.Series:
    """V字反発の取りこぼしを抑えたボラ目標エクスポージャー。

    改良点（監査・Gemini 指摘への対応）:
      1. **非対称ボラ窓**: 縮小（リスクオフ）は短窓(fast)と長窓(slow)の小さい方＝
         急上昇ボラに即応。拡大（リスクオン）は遅い長窓で慎重に戻す…のではなく、
         反発を取りこぼさないため再エントリーを別途加速（下記3）。
      2. **不感帯(dead_band)**: 目標比率の変化が dead_band 未満なら据え置き、
         日次の微小なチャーニング（往復売買コスト）を抑制。
      3. **高速再エントリー**: 市場(TOPIX)が短期トレンド転換（終値>reentry_ma日移動平均）
         したら、ボラ低下を待たずエクスポージャー下限を引き上げ、V字反発に追随。
    すべて前日までの情報で算出（ルックアヘッドなし）。
    """
    import numpy as np

    rv_slow = strat_returns.rolling(slow_window, min_periods=slow_window // 2).std() * np.sqrt(252)
    rv_fast = strat_returns.rolling(fast_window, min_periods=fast_window // 2).std() * np.sqrt(252)
    # 縮小は機敏に（高い方のボラ＝低い方の比率を採用）、復帰の取りこぼしは再エントリーで補う
    rv = pd.concat([rv_slow, rv_fast], axis=1).max(axis=1)
    raw = (target_vol / rv).clip(lower=0.0, upper=cap)

    # 高速再エントリー: TOPIX が reentry_ma 線より上なら下限 0.5 を保証
    if market_close is not None:
        m = market_close.reindex(strat_returns.index).ffill()
        above = m > m.rolling(reentry_ma, min_periods=reentry_ma // 2).mean()
        floor = above.astype(float) * 0.5
        raw = pd.concat([raw, floor], axis=1).max(axis=1).clip(upper=cap)

    raw = raw.shift(1).clip(lower=0.0).fillna(0.0)

    # 不感帯: 直近採用値から dead_band 未満の変化は無視（前方確定でルックアヘッドなし）
    out = raw.copy().to_numpy()
    last = 0.0
    for i in range(len(out)):
        if i == 0 or abs(out[i] - last) >= dead_band:
            last = out[i]
        else:
            out[i] = last
    return pd.Series(out, index=raw.index, name="vol_target_fast")


def drawdown_throttle(
    strat_returns: pd.Series,
    *,
    soft: float = -0.10,
    hard: float = -0.20,
    floor: float = 0.0,
) -> pd.Series:
    """戦略自身のドローダウンが深いほど株式比率を絞る係数(floor..1)。

    フル投資時のドローダウンを観測値として使い（前日まで）、深い下落への
    参加度を直接下げて MaxDD を抑える。soft で減らし始め、hard で floor。
    """
    eq = (1.0 + strat_returns.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    factor = ((dd - hard) / (soft - hard)).clip(lower=floor, upper=1.0)
    return factor.shift(1).fillna(1.0).rename("dd_throttle")


def trend_following_exposure(
    market_close: pd.Series,
    index: pd.DatetimeIndex,
    *,
    ma_window: int = 200,
    abs_mom_window: int = 252,
    low: float = 0.30,
    high: float = 1.0,
) -> pd.Series:
    """市場トレンドに基づく株式エクスポージャー(low..high)。

    弱気相場の現金化に実証的に堅牢なトレンドフォロー（Faber 2007; 絶対モメンタム
    Antonacci 2014）。TOPIX が「200日移動平均より上」かつ「12ヶ月リターンがプラス」
    の双方を満たせば high、いずれも欠ければ low、片方なら中間。ボラ目標と違い
    V字反発では価格が線を上抜けた時点で速やかに復帰でき取りこぼしを抑える。
    すべて前日までの情報で算出。
    """
    m = market_close.sort_index()
    ma = m.rolling(ma_window, min_periods=ma_window // 2).mean()
    above_ma = (m > ma).astype(float)
    abs_mom = (m / m.shift(abs_mom_window) - 1.0 > 0).astype(float)
    signal = (above_ma + abs_mom) / 2.0  # 0, 0.5, 1
    exposure = low + (high - low) * signal
    exposure = exposure.shift(1).reindex(index).ffill().fillna(low)
    return exposure.rename("trend_exposure")


def book_trend_exposure(
    strat_returns: pd.Series,
    *,
    ma_window: int = 200,
    low: float = 0.30,
    high: float = 1.0,
) -> pd.Series:
    """戦略ブック自身の資産曲線トレンドに基づく現金化エクスポージャー(low..high)。

    ブックの累積資産が自身の ma_window 日移動平均より上なら high、下なら low。
    市場(TOPIX)トレンドではなく**戦略固有のドローダウン局面**に同期して現金化する
    ため、Value 等の市場とズレた戦略でも有効（検証で TOPIX トレンドより優位）。
    V字反発では資産曲線が線を上抜けた時点で速やかに復帰でき取りこぼしを抑える。
    前日までの情報で算出（ルックアヘッドなし）。
    """
    eq = (1.0 + strat_returns.fillna(0.0)).cumprod()
    ma = eq.rolling(ma_window, min_periods=ma_window // 2).mean()
    signal = (eq > ma).astype(float)
    exposure = low + (high - low) * signal
    return exposure.shift(1).fillna(high).rename("book_trend")


def apply_regime_overlay(
    strat_returns: pd.Series,
    exposure: pd.Series,
    *,
    cost_bps: float = 10.0,
) -> pd.Series:
    """[簡易版] 戦略リターンにエクスポージャーを掛け、変更分のコストを引く。

    後方互換のため残置。`strat_returns` には既にリバランスコスト控除後（フル投資）の
    リターンが入る前提で、現金化変更コストのみ追加する近似。新規コードは
    `apply_overlay_full`（リバランス/現金化コストと現金金利を一体で扱う）を推奨。
    """
    exp = exposure.reindex(strat_returns.index).ffill().fillna(0.0)
    turnover = exp.diff().abs().fillna(0.0)
    overlaid = exp * strat_returns - turnover * (cost_bps * 1e-4)
    return overlaid.rename("strategy_regime")


def apply_overlay_full(
    result: BacktestResult,
    exposure: pd.Series,
    *,
    cost_bps: float = 25.0,
    cash_annual_rate: float = 0.0,
) -> pd.Series:
    """[推奨] グロス・リターンにエクスポージャーを適用する一体型オーバーレイ。

    コストの二段近似を解消し、現金部分に金利を付与する:
      - 株式部分: gross × exposure（gross は run_backtest のコスト前リターン, 廃止損込み）
      - 現金部分: (1 - exposure) × 日次現金金利（cash_annual_rate を 252 で按分）
      - コスト: 「リバランス回転 × その日の株式比率」＋「現金化エクスポージャー変更」
        を合算した片道回転に cost_bps を課金（往復は |Δ| ベースで近似）。
    exposure は前日までの情報に基づく値（各 *_exposure 関数で shift 済み）。
    """
    idx = result.gross_returns.index
    exp = exposure.reindex(idx).ffill().fillna(0.0)
    gross = result.gross_returns.fillna(0.0)
    rebal_turn = (result.rebal_turnover if result.rebal_turnover is not None
                  else pd.Series(0.0, index=idx)).reindex(idx).fillna(0.0)

    cash_daily = cash_annual_rate / 252.0
    cost_rate = cost_bps * 1e-4

    # リバランス由来コスト: 銘柄入替の片道回転(rebal_turn)に当日の株式比率を掛ける
    # （現金化中は実際の売買額が比率ぶん小さい）。往復近似で *2。
    rebal_cost = rebal_turn * exp * cost_rate * 2.0
    # 現金化由来コスト: エクスポージャー変更分（株を売って現金へ/その逆）
    exp_change_cost = exp.diff().abs().fillna(0.0) * cost_rate

    net = exp * gross + (1.0 - exp) * cash_daily - rebal_cost - exp_change_cost
    return net.rename("strategy_overlay")
