"""日次の売買シグナル生成（手動執行向けの推奨ポートフォリオ）。

最新営業日のマルチファクター・スコアで目標銘柄を選び、現金化エクスポージャー
（ボラ目標）を反映して、資金額に対する目標保有（単元=100株単位）を算出する。
出力は reports/daily_signal.md。実際の発注は本人が証券会社画面で行う。

使い方:
  python -m quant_jp.ops.daily_report --capital 3000000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_jp.backtest.engine import book_trend_exposure, run_backtest
from quant_jp.data import load, universe
from quant_jp.strategy import ranking

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"
LOT = 100


def generate(capital: float = 3_000_000, top_n: int = 20, cost_bps: float = 25.0) -> str:
    close = load.close_panel()
    listed = load.load_listed().set_index("Code")
    eligible = universe.eligible_mask().reindex(
        index=close.index, columns=close.columns
    ).fillna(False)

    # バリュー主軸・スタッガード(3トランチ)。現金化はブック200日トレンド。
    score = ranking.value_tilt_score(close, eligible, quality_w=0.0)
    weights = ranking.select_weights_staggered(
        close, eligible, top_n=top_n, exit_n=2 * top_n,
        score=score, use_trend_filter=False,
    )
    res_inv = run_backtest(close, weights, exposure=None, rebalance="ME", cost_bps=cost_bps)
    exposure = book_trend_exposure(res_inv.gross_returns, ma_window=200, low=0.30)

    asof = close.index[-1]
    w = weights.loc[asof]
    w = w[w > 0].sort_values(ascending=False)
    exp = float(exposure.loc[asof])
    px = close.loc[asof]

    rows = []
    for code, wt in w.items():
        price = px.get(code, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        # 丸め超過で資金を超えないよう余裕(0.99)を見て切り捨て
        target_yen = capital * wt * exp * 0.99
        # 単元未満株（S株/かぶミニ）前提＝1株単位で比率に忠実に保有
        shares = int(target_yen / price)
        lot100 = int(round(target_yen / (price * LOT))) * LOT  # 単元株運用時の参考
        name = str(listed["CoName"].get(code, ""))[:20] if not listed.empty else ""
        rows.append((code, name, wt, price, shares, shares * price, lot100))

    df = pd.DataFrame(rows, columns=["Code", "銘柄", "比率", "株価", "株数", "金額", "単元株参考"])
    invested = df["金額"].sum()
    cash = capital - invested

    lines = [
        "# 日次売買シグナル（推奨ポートフォリオ）",
        "",
        f"- 基準日: {asof.date()}（この終値ベース。翌営業日に発注を想定）",
        f"- 資金: {capital:,.0f} 円 / 推奨株式エクスポージャー: **{exp:.0%}**（残りは現金）",
        f"- シグナル: バリュー主軸 E/P・B/P（逆ボラ加重, 上位{top_n}銘柄, 四半期）",
        f"- 株式投資額: {invested:,.0f} 円 / 現金: {cash:,.0f} 円",
        "",
        "| コード | 銘柄 | 比率 | 株価 | 株数(1株単位) | 概算金額 | 単元株参考 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Code']} | {r['銘柄']} | {r['比率']:.1%} | {r['株価']:,.0f} | "
            f"{r['株数']:,} | {r['金額']:,.0f} | {r['単元株参考']:,} |"
        )
    lines += [
        "",
        f"> 株数は**単元未満株（SBI S株 / 楽天かぶミニ）前提の1株単位**で、比率に忠実。",
        "> 「単元株参考」は100株単位で運用する場合の目安（端数で比率がぶれる）。",
        f"> エクスポージャー {exp:.0%} は戦略ブックの200日トレンドが崩れると自動的に下がり現金化が進む。",
    ]
    report = "\n".join(lines)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "daily_signal.md").write_text(report + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=3_000_000)
    ap.add_argument("--top_n", type=int, default=20)
    ap.add_argument("--cost_bps", type=float, default=25.0)
    args = ap.parse_args()
    report = generate(args.capital, args.top_n, args.cost_bps)
    print(report)
    print(f"\n保存: {REPORTS / 'daily_signal.md'}")


if __name__ == "__main__":
    main()
