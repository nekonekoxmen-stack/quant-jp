"""メイン戦略のエンドツーエンド・バックテスト。

データ → 流動性ユニバース(PIT) → バリュー主軸シグナル(逆ボラ加重・四半期バッファ) →
ブック・トレンド現金化オーバーレイ → コスト込みバックテスト → 対 TOPIX 比較。

検証の結論（AUDIT.md）に基づく構成:
  - モメンタムは現実的コスト下でエッジが消えるため不採用。バリューが日本株で最も頑健。
  - 現金化は「戦略ブック自身の200日トレンド」。TOPIX トレンドやボラ目標より優位。

使い方:
  python -m quant_jp.backtest.run_baseline
  python -m quant_jp.backtest.run_baseline --top_n 20 --cost_bps 25 --rebalance Q
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quant_jp.backtest import metrics  # noqa: E402
from quant_jp.backtest.engine import (  # noqa: E402
    apply_overlay_full,
    book_trend_exposure,
    run_backtest,
    topix_returns,
    vol_target_exposure,
)
from quant_jp.backtest.walkforward import bootstrap_ci, deflated_sharpe, stitch_oos  # noqa: E402
from quant_jp.data import load, universe  # noqa: E402
from quant_jp.strategy import ranking  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_n", type=int, default=20)
    # 片道コスト = 手数料 + 単元未満株スプレッド + スリッページ。S株/かぶミニの
    # 実勢スプレッド(片道0.2〜0.5%)を踏まえ保守的な既定25bps。
    ap.add_argument("--cost_bps", type=float, default=25.0)
    ap.add_argument("--rebalance", default="Q", help="リバランス頻度 (Q=四半期, ME=月次)")
    ap.add_argument("--min_adv_yen", type=float, default=1e8)
    ap.add_argument("--max_lot_cost_yen", type=float, default=1e6)
    ap.add_argument("--exit_n", type=int, default=40, help="バッファ: この順位以下で売却")
    ap.add_argument("--quality_w", type=float, default=0.0, help="クオリティの加味度(0-1)")
    ap.add_argument("--cashout", default="book_trend", choices=["book_trend", "vol", "none"])
    ap.add_argument("--cash_rate", type=float, default=0.005, help="現金部分の年率金利(既定0.5%)")
    ap.add_argument("--staggered", action="store_true", default=True, help="スタッガード(3トランチ)")
    ap.add_argument("--no-staggered", dest="staggered", action="store_false")
    ap.add_argument("--n_trials", type=int, default=20, help="Deflated Sharpe 用の試行数")
    ap.add_argument("--start", default=None, help="バックテスト開始日 YYYY-MM-DD")
    args = ap.parse_args()

    print("データ読込 ...", flush=True)
    close = load.close_panel()
    topix = load.load_topix().set_index("Date")["Close"]
    if close.empty:
        raise SystemExit("日足データがありません。先に ingest pull を実行してください。")

    eligible = universe.eligible_mask(
        min_adv_yen=args.min_adv_yen, max_lot_cost_yen=args.max_lot_cost_yen
    )
    eligible = eligible.reindex(index=close.index, columns=close.columns).fillna(False)

    print("シグナル生成（バリュー主軸） ...", flush=True)
    score = ranking.value_tilt_score(close, eligible, quality_w=args.quality_w)
    if args.staggered:
        weights = ranking.select_weights_staggered(
            close, eligible, top_n=args.top_n, exit_n=args.exit_n,
            score=score, use_trend_filter=False,
        )
        rebal_for_bt = "ME"  # スタッガードは毎月入替が起きるので月次でコスト評価
    else:
        weights = ranking.select_weights_buffered(
            close, eligible, top_n=args.top_n, exit_n=args.exit_n, rebalance=args.rebalance,
            score=score, use_trend_filter=False,
        )
        rebal_for_bt = args.rebalance

    warmup = 252
    start = close.index[warmup] if args.start is None else args.start

    print("バックテスト（一体型コスト＋現金金利） ...", flush=True)
    res_inv = run_backtest(
        close, weights, exposure=None, rebalance=rebal_for_bt, cost_bps=args.cost_bps
    )
    # 現金化エクスポージャー（前日まで情報, gross ベースで判定）
    if args.cashout == "book_trend":
        exposure = book_trend_exposure(res_inv.gross_returns, ma_window=200, low=0.30)
    elif args.cashout == "vol":
        exposure = vol_target_exposure(res_inv.gross_returns, target_vol=0.15)
    else:
        exposure = res_inv.gross_returns * 0 + 1.0

    # 一体型オーバーレイ: コスト二段近似を解消＋現金部分に金利
    r_regime = apply_overlay_full(
        res_inv, exposure, cost_bps=args.cost_bps, cash_annual_rate=args.cash_rate
    )
    # 現金化なし版（フル投資, コスト込み）も一体型で算出
    full_exp = pd.Series(1.0, index=res_inv.gross_returns.index)
    r_inv = apply_overlay_full(res_inv, full_exp, cost_bps=args.cost_bps, cash_annual_rate=args.cash_rate)

    bench = topix_returns(topix).reindex(close.index).fillna(0.0)
    r_full = r_regime.loc[start:]
    r_noreg = r_inv.loc[start:]
    r_bench = bench.loc[start:]

    s_full = metrics.summary(r_full, benchmark=r_bench)
    s_noreg = metrics.summary(r_noreg, benchmark=r_bench)
    s_bench = metrics.summary(r_bench)

    # --- walk-forward OOS（本番の主指標として併記） ---
    print("Walk-forward OOS 評価 ...", flush=True)
    oos = stitch_oos(r_regime, close.index, train_years=3, test_years=1)
    oos_b = bench.reindex(oos.index)
    s_oos = metrics.summary(oos, benchmark=oos_b)
    ci = bootstrap_ci(r_full, r_bench)
    dsr = deflated_sharpe(r_full, n_trials=args.n_trials)

    rebal_label = "スタッガード(3トランチ,毎月1/3)" if args.staggered else args.rebalance
    lines = [
        "# メイン戦略バックテスト結果（バリュー主軸＋ブック・トレンド現金化）",
        "",
        f"- 期間: {r_full.index[0].date()} 〜 {r_full.index[-1].date()}",
        f"- 設定: top_n={args.top_n}, cost_bps={args.cost_bps}(片道), rebalance={rebal_label}, "
        f"exit_n={args.exit_n}, quality_w={args.quality_w}, cashout={args.cashout}, "
        f"cash_rate={args.cash_rate:.1%}",
        "- シグナル: バリュー(E/P・B/P)主軸・逆ボラ加重／現金化: ブック200日トレンド／"
        "コスト・現金金利を一体評価",
        "",
        "| 戦略 | " + " | ".join(s_full.keys()) + " |",
        "|" + "---|" * (len(s_full) + 1),
        _row("バリュー＋現金化(全期間IS)", s_full),
        _row("バリュー(現金化なし)", s_noreg),
        _row("TOPIX", s_bench),
        "",
        "## Walk-forward OOS（過剰最適化の検証, 学習3年→検証1年）",
        "",
        f"- OOS連結: CAGR {s_oos['CAGR']:.2%} / Sharpe {s_oos['Sharpe']:.2f} / "
        f"MaxDD {s_oos['MaxDD']:.2%} / 対TOPIX超過 {s_oos['Excess_CAGR']:.2%}",
        f"  （期間 {oos.index[0].date()}〜{oos.index[-1].date()}）",
        "",
        "## 統計的有意性",
        "",
        f"- 超過CAGR: 平均 {ci['excess_cagr_mean']:.2%}, 95%CI [{ci['excess_cagr_ci'][0]:.2%}, "
        f"{ci['excess_cagr_ci'][1]:.2%}], p値(<=0) {ci['excess_cagr_pval']:.3f}",
        f"- Sharpe差 p値(<=0) {ci['sharpe_diff_pval']:.3f}",
        f"- Deflated Sharpe（試行{args.n_trials}回補正後の真Sharpe>0確率）: {dsr:.3f}",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "baseline_result.md").write_text(report + "\n", encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[3, 1])
    metrics.equity_curve(r_full).plot(ax=ax1, label="Value + Cash-out")
    metrics.equity_curve(r_noreg).plot(ax=ax1, label="Value (full invest)")
    metrics.equity_curve(r_bench).plot(ax=ax1, label="TOPIX", linestyle="--")
    ax1.set_yscale("log")
    ax1.set_title("Cumulative equity (log scale)")
    ax1.legend()
    exposure.loc[start:].plot(ax=ax2, color="tab:green")
    ax2.set_title("Equity exposure (book-trend cash-out)")
    ax2.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(REPORTS / "baseline_equity.png", dpi=110)
    print(f"\n保存: {REPORTS / 'baseline_result.md'} / {REPORTS / 'baseline_equity.png'}")


def _row(name: str, s: dict) -> str:
    pct = {"CAGR", "AnnVol", "MaxDD", "Bench_CAGR", "Bench_MaxDD", "Excess_CAGR"}
    cells = [f"{v:.2%}" if k in pct else f"{v:.2f}" for k, v in s.items()]
    return f"| {name} | " + " | ".join(cells) + " |"


if __name__ == "__main__":
    main()
