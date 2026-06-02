"""ペーパートレード（仮想運用）台帳。

固定の開始日・初期資金から本番戦略（バリュー主軸＋スタッガード＋ブック200日トレンド
現金化）を前進シミュレーションし、仮想口座の日次資産・現在保有・対TOPIX 成績を出力。

状態を外部ファイルに持ち回ると破損・不整合のリスクがあるため、**毎回データ先頭から
決定論的に再計算**する方式。新しい営業日のデータが増えれば自動的に評価期間が伸びる。
日々の推奨スナップショットは監査用に CSV へ追記保存もする（再現性確認用）。

使い方:
  python -m quant_jp.ops.paper_trade            # 台帳更新＋サマリ表示
  python -m quant_jp.ops.paper_trade --start 2026-06-01 --capital 1500000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"

# --- ペーパー運用の固定パラメータ（実運用と同一） ---
START_DATE = "2026-06-01"   # 仮想運用の開始日
INIT_CAPITAL = 1_500_000    # 初期資金（円）
TOP_N = 20
COST_BPS = 25.0
CASH_RATE = 0.005           # 現金部分の年率金利
BOOK_TREND_MA = 200
EXPOSURE_LOW = 0.30


@dataclass
class PaperResult:
    start: pd.Timestamp
    asof: pd.Timestamp
    init_capital: float
    equity: pd.Series          # 仮想口座の資産推移（円）
    topix_equity: pd.Series     # 同額をTOPIXに投じた場合
    exposure: pd.Series         # 株式比率の推移
    holdings: pd.DataFrame      # 現在の推奨保有（コード/銘柄/比率/株価/株数/金額）
    summary: dict               # 期間成績（戦略・TOPIX・超過）
    invested_yen: float
    cash_yen: float
    exposure_now: float


def run_paper(start: str = START_DATE, capital: float = INIT_CAPITAL,
              top_n: int = TOP_N) -> PaperResult:
    close = load.close_panel()
    topix = load.load_topix().set_index("Date")["Close"]
    listed = load.load_listed().set_index("Code")
    eligible = universe.eligible_mask().reindex(
        index=close.index, columns=close.columns
    ).fillna(False)

    # 本番と同一のシグナル・現金化（全期間で計算し、開始日以降を口座成績として切出す）
    score = ranking.value_tilt_score(close, eligible, quality_w=0.0)
    weights = ranking.select_weights_staggered(
        close, eligible, top_n=top_n, exit_n=2 * top_n,
        score=score, use_trend_filter=False,
    )
    res_inv = run_backtest(close, weights, exposure=None, rebalance="ME", cost_bps=COST_BPS)
    exposure = book_trend_exposure(res_inv.gross_returns, ma_window=BOOK_TREND_MA, low=EXPOSURE_LOW)
    daily_ret = apply_overlay_full(res_inv, exposure, cost_bps=COST_BPS, cash_annual_rate=CASH_RATE)
    bench_ret = topix_returns(topix).reindex(close.index).fillna(0.0)

    start_ts = pd.Timestamp(start)
    idx = close.index[close.index >= start_ts]
    if len(idx) == 0:
        # まだ開始日に達していない（=運用前）。直近日をasofに、資産は初期額のまま。
        asof = close.index[-1]
        empty = pd.Series([capital], index=[asof])
        holds = _current_holdings(weights, exposure, close, listed, capital, asof)
        return PaperResult(
            start=start_ts, asof=asof, init_capital=capital,
            equity=empty, topix_equity=empty.copy(),
            exposure=pd.Series([float(exposure.loc[asof])], index=[asof]),
            holdings=holds[0], summary={}, invested_yen=holds[1], cash_yen=holds[2],
            exposure_now=float(exposure.loc[asof]),
        )

    r = daily_ret.reindex(idx).fillna(0.0)
    b = bench_ret.reindex(idx).fillna(0.0)
    # 初日は建玉を組成する日（終値で取得）とし、P&L=0 から開始する。
    # これにより初日のレジーム変更コスト等による見かけのマイナスを除く。
    if len(r) >= 1:
        r.iloc[0] = 0.0
        b.iloc[0] = 0.0
    equity = capital * (1.0 + r).cumprod()
    topix_equity = capital * (1.0 + b).cumprod()
    exp_path = exposure.reindex(idx).ffill().fillna(EXPOSURE_LOW)

    asof = idx[-1]
    holds_df, invested, cash = _current_holdings(weights, exposure, close, listed, capital, asof)

    # サンプルが少ないうちは Sharpe/年率指標が無意味（nan）になるため抑制。
    # 実評価日数が MIN_EVAL_DAYS 未満なら「成績ウォームアップ中」として総リターンのみ表示。
    MIN_EVAL_DAYS = 20
    n_days = len(r)
    summ = {
        "FinalEquity": float(equity.iloc[-1]),
        "TopixEquity": float(topix_equity.iloc[-1]),
        "TotalReturn": float(equity.iloc[-1] / capital - 1.0),
        "TopixReturn": float(topix_equity.iloc[-1] / capital - 1.0),
        "NDays": n_days,
        "Warmup": n_days < MIN_EVAL_DAYS,
    }
    if n_days >= MIN_EVAL_DAYS:
        full = metrics.summary(r, benchmark=b)
        summ.update(full)

    return PaperResult(
        start=start_ts, asof=asof, init_capital=capital,
        equity=equity, topix_equity=topix_equity, exposure=exp_path,
        holdings=holds_df, summary=summ,
        invested_yen=invested, cash_yen=cash, exposure_now=float(exp_path.iloc[-1]),
    )


def _current_holdings(weights, exposure, close, listed, capital, asof):
    """asof 時点の推奨保有（目標株数）を返す。

    株価・株数は**生（無調整）株価**で算出する。close_panel は分割調整後のため、
    分割履歴のある銘柄では実際の発注価格とズレ、誤った株数になる。発注は実価格で
    行うので、表示も実価格に統一する。
    """
    w = weights.loc[asof]
    w = w[w > 0].sort_values(ascending=False)
    exp = float(exposure.loc[asof])
    raw_px = load.raw_close_panel()
    raw_row = raw_px.loc[asof] if asof in raw_px.index else None
    adj_row = close.loc[asof]
    rows = []
    for code, wt in w.items():
        # 実発注用は生株価。欠損時は調整後で代替（稀）。
        price = raw_row.get(code, np.nan) if raw_row is not None else np.nan
        if not np.isfinite(price) or price <= 0:
            price = adj_row.get(code, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        target_yen = capital * wt * exp * 0.99
        shares = int(target_yen / price)
        name = str(listed["CoName"].get(code, "")) if not listed.empty else ""
        rows.append({"コード": code, "銘柄": name, "比率": float(wt),
                     "株価": float(price), "株数": shares, "金額": shares * price})
    df = pd.DataFrame(rows)
    invested = float(df["金額"].sum()) if not df.empty else 0.0
    return df, invested, capital - invested


def append_snapshot(result: PaperResult) -> None:
    """当日の推奨スナップショットを台帳CSVへ追記（同日重複は上書き）。"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    ledger = REPORTS / "paper_ledger.csv"
    snap = result.holdings.copy()
    if snap.empty:
        return
    snap.insert(0, "日付", result.asof.date())
    snap.insert(1, "株式比率", round(result.exposure_now, 4))
    if ledger.exists():
        old = pd.read_csv(ledger)
        old = old[old["日付"] != str(result.asof.date())]  # 同日分を除去して入れ直す
        snap = pd.concat([old, snap], ignore_index=True)
    snap.to_csv(ledger, index=False, encoding="utf-8-sig")


def _fmt(result: PaperResult) -> str:
    s = result.summary
    lines = [
        "# ペーパートレード（仮想運用）サマリ",
        "",
        f"- 開始日: {result.start.date()} / 初期資金: {result.init_capital:,.0f} 円",
        f"- 基準日: {result.asof.date()}",
    ]
    if s and not s.get("Warmup", False):
        lines += [
            f"- 仮想口座 評価額: **{s['FinalEquity']:,.0f} 円**（{s['TotalReturn']:+.2%}）",
            f"- 同額TOPIX: {s['TopixEquity']:,.0f} 円（{s['TopixReturn']:+.2%}）",
            f"- 対TOPIX超過(年率): {s['Excess_CAGR']:+.2%} / Sharpe {s['Sharpe']:.2f} / "
            f"最大DD {s['MaxDD']:.2%}",
        ]
    elif s:
        lines += [
            f"- 仮想口座 評価額: **{s['FinalEquity']:,.0f} 円**（{s['TotalReturn']:+.2%}）",
            f"- 同額TOPIX: {s['TopixEquity']:,.0f} 円（{s['TopixReturn']:+.2%}）",
            f"- 運用 {s.get('NDays', 0)} 日目（20日でSharpeなどリスクKPIの集計を開始）",
        ]
    else:
        lines.append("- （運用開始日に未達。開始日以降に成績が記録されます）")
    lines += [
        f"- 現在の株式比率: {result.exposure_now:.0%}（株式 {result.invested_yen:,.0f} 円 / "
        f"現金 {result.cash_yen:,.0f} 円）",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START_DATE)
    ap.add_argument("--capital", type=float, default=INIT_CAPITAL)
    ap.add_argument("--top_n", type=int, default=TOP_N)
    args = ap.parse_args()
    result = run_paper(args.start, args.capital, args.top_n)
    append_snapshot(result)
    report = _fmt(result)
    print(report)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "paper_summary.md").write_text(report + "\n", encoding="utf-8")
    print(f"\n台帳: {REPORTS / 'paper_ledger.csv'} / サマリ: {REPORTS / 'paper_summary.md'}")


if __name__ == "__main__":
    main()
