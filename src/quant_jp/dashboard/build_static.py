"""静的HTMLダッシュボードを生成（GitHub Pages 公開用）。

Streamlit と異なりサーバ不要で、開いた瞬間に表示される。GitHub Actions が毎営業日
これを再生成して Pages へデプロイする。デザインは Streamlit 版（Google 風・白基調）を踏襲。

出力: site/index.html（Plotly を inline 埋め込み、外部依存は CDN の plotly.js のみ）
使い方: python -m quant_jp.dashboard.build_static
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
from plotly.io import to_html

from quant_jp.backtest import metrics
from quant_jp.ops.paper_trade import run_paper

ROOT = Path(__file__).resolve().parents[3]
SITE = ROOT / "site"

BLUE, GREEN, GREY, YELLOW = "#1a73e8", "#34a853", "#5f6368", "#fbbc04"
BORDER = "#dadce0"

PLOT_LAYOUT = dict(
    font=dict(family="Roboto, sans-serif", color="#202124", size=13),
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    xaxis=dict(gridcolor="#f1f3f4"), yaxis=dict(gridcolor="#f1f3f4"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    title=dict(x=0, xanchor="left", font=dict(size=15, color="#202124")),
    margin=dict(t=70, b=20, l=10, r=10),
)


def _kpi(label: str, value: str, delta: str = "", cls: str = "neutral") -> str:
    d = f'<div class="kpi-delta kpi-{cls}">{delta}</div>' if delta else ""
    return (f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{d}</div>')


def _fig_html(fig: go.Figure, div_id: str, include_js: bool) -> str:
    return to_html(fig, include_plotlyjs="cdn" if include_js else False,
                   full_html=False, div_id=div_id, config={"displayModeBar": False})


def build() -> Path:
    res = run_paper()
    has_perf = bool(res.summary)

    # KPI
    if has_perf:
        s = res.summary
        kpis = (
            _kpi("評価額", f"¥{s['FinalEquity']:,.0f}",
                 f"{s['TotalReturn']:+.1%}", "up" if s["TotalReturn"] >= 0 else "down")
            + _kpi("対TOPIX超過(年率)", f"{s['Excess_CAGR']:+.1%}", "",
                   "up" if s["Excess_CAGR"] >= 0 else "down")
            + _kpi("Sharpe", f"{s['Sharpe']:.2f}", "", "neutral")
            + _kpi("最大DD", f"{s['MaxDD']:.1%}", "", "neutral")
            + _kpi("株式比率", f"{res.exposure_now:.0%}", "残りは現金", "neutral")
        )
    else:
        kpis = (
            _kpi("状態", "運用開始前", f"{res.start.date()} 開始予定", "neutral")
            + _kpi("初期資金", f"¥{res.init_capital:,.0f}", "", "neutral")
            + _kpi("株式比率(予定)", f"{res.exposure_now:.0%}", "", "neutral")
        )

    # チャート
    charts_html = ""
    first = True
    if has_perf and len(res.equity) > 1:
        fig = go.Figure()
        fig.add_scatter(x=res.equity.index, y=res.equity.values, name="仮想口座",
                        line=dict(color=BLUE, width=2.5))
        fig.add_scatter(x=res.topix_equity.index, y=res.topix_equity.values, name="TOPIX",
                        line=dict(color=GREY, width=1.5, dash="dash"))
        fig.update_layout(height=420, title="仮想口座の資産推移（円）", **PLOT_LAYOUT)
        charts_html += f'<div class="card">{_fig_html(fig, "eq", first)}</div>'
        first = False

        fig2 = go.Figure()
        fig2.add_scatter(x=res.exposure.index, y=res.exposure.values, name="株式比率",
                         fill="tozeroy", line=dict(color=GREEN, width=1.5),
                         fillcolor="rgba(52,168,83,.12)")
        fig2.update_yaxes(range=[0, 1.05], tickformat=".0%")
        fig2.update_layout(height=220, title="株式エクスポージャー（現金化ダイヤル）", **PLOT_LAYOUT)
        charts_html += f'<div class="card">{_fig_html(fig2, "exp", first)}</div>'
        first = False

    # 保有テーブル
    rows = ""
    for _, r in res.holdings.iterrows():
        rows += (f"<tr><td>{r['コード']}</td><td>{r['銘柄']}</td>"
                 f"<td class='num'>{r['比率']*100:.1f}%</td>"
                 f"<td class='num'>¥{r['株価']:,.0f}</td>"
                 f"<td class='num'>{int(r['株数']):,}</td>"
                 f"<td class='num'>¥{r['金額']:,.0f}</td></tr>")
    table = (
        "<table class='holdings'><thead><tr><th>コード</th><th>銘柄</th>"
        "<th class='num'>比率</th><th class='num'>株価</th><th class='num'>株数</th>"
        "<th class='num'>概算金額</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )

    html = _PAGE.format(
        asof=res.asof.date(), start=res.start.date(),
        kpis=kpis, charts=charts_html, table=table,
        invested=f"{res.invested_yen:,.0f}", cash=f"{res.cash_yen:,.0f}",
        exp=f"{res.exposure_now:.0%}",
    )
    SITE.mkdir(parents=True, exist_ok=True)
    out = SITE / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


_PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>日本株クオンツ ペーパートレード</title>
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family:'Roboto',-apple-system,'Segoe UI',sans-serif; color:#202124;
          background:#ffffff; margin:0; padding:24px; max-width:1100px; margin:0 auto; }}
  .header {{ display:flex; align-items:baseline; gap:12px; padding-bottom:10px;
             border-bottom:1px solid #dadce0; margin-bottom:20px; }}
  .title {{ font-size:26px; font-weight:500; letter-spacing:-.3px; }}
  .sub {{ font-size:13px; color:#5f6368; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
               gap:14px; margin-bottom:20px; }}
  .kpi-card {{ background:#fff; border:1px solid #dadce0; border-radius:12px; padding:16px 18px;
               box-shadow:0 1px 2px rgba(60,64,67,.08); }}
  .kpi-label {{ font-size:12px; color:#5f6368; font-weight:500; text-transform:uppercase; letter-spacing:.3px; }}
  .kpi-value {{ font-size:26px; font-weight:500; margin-top:4px; }}
  .kpi-delta {{ font-size:12px; margin-top:6px; font-weight:500; }}
  .kpi-up {{ color:#34a853; }} .kpi-down {{ color:#ea4335; }} .kpi-neutral {{ color:#5f6368; }}
  .card {{ background:#fff; border:1px solid #dadce0; border-radius:12px; padding:12px 16px;
           margin-bottom:16px; box-shadow:0 1px 2px rgba(60,64,67,.08); }}
  .section {{ font-size:16px; font-weight:500; margin:18px 0 8px; }}
  .chip {{ display:inline-block; padding:4px 12px; border-radius:16px; font-size:13px;
           font-weight:500; background:#e8f0fe; color:#1a73e8; margin-right:8px; }}
  table.holdings {{ width:100%; border-collapse:collapse; font-size:14px; }}
  table.holdings th {{ text-align:left; color:#5f6368; font-weight:500; padding:8px 10px;
                       border-bottom:1px solid #dadce0; font-size:12px; }}
  table.holdings td {{ padding:8px 10px; border-bottom:1px solid #f1f3f4; }}
  table.holdings td.num, table.holdings th.num {{ text-align:right; }}
  .footer {{ color:#5f6368; font-size:12px; margin-top:20px; }}
</style></head>
<body>
  <div class="header"><span class="title">日本株クオンツ</span>
    <span class="sub">ペーパートレード ・ 基準日 {asof} ・ 開始 {start}</span></div>
  <div class="kpi-grid">{kpis}</div>
  {charts}
  <div class="section">推奨ポートフォリオ（単元未満株＝1株単位）</div>
  <div><span class="chip">株式 ¥{invested}</span><span class="chip">現金 ¥{cash}</span>
       <span class="chip">株式比率 {exp}</span></div>
  <div class="card" style="margin-top:10px">{table}</div>
  <div class="footer">バリュー主軸（E/P・B/P）＋ブック200日トレンド現金化／コスト25bps・現金金利0.5%控除後。
    仮想運用であり実際の売買・利益を保証するものではありません。</div>
</body></html>
"""


def main() -> None:
    out = build()
    print(f"生成: {out}")


if __name__ == "__main__":
    main()
