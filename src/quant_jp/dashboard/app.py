"""日本株クオンツ・ダッシュボード（Streamlit / Google 風・白基調）。

起動:
  uv run streamlit run src/quant_jp/dashboard/app.py
スマホ等から見る場合は --server.address 0.0.0.0 を付けて同一LANからアクセス。
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from quant_jp.dashboard.data import build

st.set_page_config(page_title="日本株クオンツ", layout="wide", page_icon="📊")

# --- Google 風スタイル（Material Design ライク・白基調） ---
GOOGLE_BLUE = "#1a73e8"
GOOGLE_GREEN = "#34a853"
GOOGLE_RED = "#ea4335"
GOOGLE_YELLOW = "#fbbc04"
GREY = "#5f6368"
BORDER = "#dadce0"

st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
      html, body, [class*="css"] {{
        font-family: 'Roboto', -apple-system, 'Segoe UI', sans-serif;
        color: #202124;
      }}
      .stApp {{ background-color: #ffffff; }}
      /* ヘッダーバー */
      .gh-header {{
        display: flex; align-items: baseline; gap: 12px;
        padding: 8px 0 4px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 20px;
      }}
      .gh-title {{ font-size: 26px; font-weight: 500; color: #202124; letter-spacing: -0.3px; }}
      .gh-sub {{ font-size: 13px; color: {GREY}; }}
      /* KPI カード */
      .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin: 8px 0 18px 0; }}
      .kpi-card {{
        background: #ffffff; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 16px 18px; box-shadow: 0 1px 2px rgba(60,64,67,.08);
        transition: box-shadow .2s;
      }}
      .kpi-card:hover {{ box-shadow: 0 1px 6px rgba(60,64,67,.2); }}
      .kpi-label {{ font-size: 12px; color: {GREY}; font-weight: 500; letter-spacing: .3px; text-transform: uppercase; }}
      .kpi-value {{ font-size: 28px; font-weight: 500; color: #202124; margin-top: 4px; line-height: 1.1; }}
      .kpi-delta {{ font-size: 12px; margin-top: 6px; font-weight: 500; }}
      .kpi-up {{ color: {GOOGLE_GREEN}; }}
      .kpi-down {{ color: {GOOGLE_RED}; }}
      .kpi-neutral {{ color: {GREY}; }}
      /* セクション見出し */
      .gh-section {{ font-size: 16px; font-weight: 500; color: #202124; margin: 18px 0 8px 0; }}
      /* チップ（株式比率バッジ） */
      .gh-chip {{
        display: inline-block; padding: 4px 12px; border-radius: 16px;
        font-size: 13px; font-weight: 500; background: #e8f0fe; color: {GOOGLE_BLUE};
      }}
      /* タブ */
      .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
      .stTabs [data-baseweb="tab"] {{
        font-weight: 500; color: {GREY}; padding: 8px 16px;
      }}
      .stTabs [aria-selected="true"] {{ color: {GOOGLE_BLUE}; }}
      /* サイドバー */
      section[data-testid="stSidebar"] {{ background: #f8f9fa; border-right: 1px solid {BORDER}; }}
      #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    font=dict(family="Roboto, sans-serif", color="#202124", size=13),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    xaxis=dict(gridcolor="#f1f3f4", zerolinecolor="#dadce0"),
    yaxis=dict(gridcolor="#f1f3f4", zerolinecolor="#dadce0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(t=30, b=20, l=10, r=10),
)


@st.cache_data(show_spinner="戦略を計算中...")
def _load(capital: float, top_n: int, quality_w: float, cost_bps: float, rebalance: str):
    return build(capital, top_n, quality_w, cost_bps, rebalance)


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _kpi(label: str, value: str, delta: str = "", cls: str = "neutral") -> str:
    d = f'<div class="kpi-delta kpi-{cls}">{delta}</div>' if delta else ""
    return (
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{d}</div>'
    )


# --- サイドバー（設定） ---
st.sidebar.markdown("### ⚙️ 設定")
capital = st.sidebar.number_input("資金（円）", value=3_000_000, step=100_000, min_value=100_000)
top_n = st.sidebar.slider("保有銘柄数", 10, 40, 20)
quality_w = st.sidebar.slider("クオリティ加味度", 0.0, 0.5, 0.0, 0.05)
cost_bps = st.sidebar.slider("取引コスト（bps, 片道）", 0.0, 60.0, 25.0, 1.0)
rebalance = st.sidebar.selectbox("リバランス", ["Q", "ME"], index=0)
st.sidebar.caption("バリュー主軸＋ブック200日トレンド現金化")

d = _load(capital, top_n, quality_w, cost_bps, rebalance)
s = d.summary_strategy

# --- ヘッダー ---
st.markdown(
    f'<div class="gh-header"><span class="gh-title">日本株クオンツ</span>'
    f'<span class="gh-sub">基準日 {d.asof.date()} ・ バリュー × トレンド現金化</span></div>',
    unsafe_allow_html=True,
)

# --- KPI カード ---
exc = s["Excess_CAGR"]
dd_diff = s["MaxDD"] - d.summary_topix["MaxDD"]
cards = (
    _kpi("CAGR", _pct(s["CAGR"]), f"{'+' if exc>=0 else ''}{_pct(exc)} vs TOPIX",
         "up" if exc >= 0 else "down")
    + _kpi("Sharpe", f"{s['Sharpe']:.2f}", f"TOPIX {d.summary_topix['Sharpe']:.2f}", "neutral")
    + _kpi("最大ドローダウン", _pct(s["MaxDD"]),
           f"{'+' if dd_diff>=0 else ''}{_pct(dd_diff)} vs TOPIX", "up" if dd_diff >= 0 else "down")
    + _kpi("年率ボラティリティ", _pct(s["AnnVol"]), "", "neutral")
    + _kpi("現在の株式比率", _pct(d.exposure_now), "残りは現金", "neutral")
)
st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["今日の推奨", "資産曲線", "年次・現金化"])

with tab1:
    st.markdown('<div class="gh-section">推奨ポートフォリオ（単元未満株＝1株単位）</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<span class="gh-chip">株式 {d.invested_yen:,.0f} 円</span>&nbsp;&nbsp;'
        f'<span class="gh-chip">現金 {d.cash_yen:,.0f} 円</span>&nbsp;&nbsp;'
        f'<span class="gh-chip">株式比率 {d.exposure_now:.0%}</span>',
        unsafe_allow_html=True,
    )
    st.write("")
    show = d.portfolio.copy()
    if "比率" in show.columns:
        show["比率"] = show["比率"] * 100.0  # 小数→% 表示用
    st.dataframe(
        show, use_container_width=True, hide_index=True,
        column_config={
            "比率": st.column_config.NumberColumn("比率(%)", format="%.1f"),
            "株価": st.column_config.NumberColumn("株価", format="¥%d"),
            "株数": st.column_config.NumberColumn("株数", format="%d"),
            "金額": st.column_config.NumberColumn("概算金額", format="¥%d"),
        },
    )
    st.caption("実際の発注はご自身で証券会社にて。弱気局面では株式比率が自動的に下がり現金化が進みます。")

with tab2:
    st.markdown('<div class="gh-section">累積資産（評価期間・コスト後）</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_scatter(x=d.equity.index, y=d.equity["strategy"], name="戦略（現金化あり）",
                    line=dict(color=GOOGLE_BLUE, width=2.5))
    fig.add_scatter(x=d.equity.index, y=d.equity["full"], name="戦略（フル投資）",
                    line=dict(color=GOOGLE_YELLOW, width=1.5, dash="dot"))
    fig.add_scatter(x=d.equity.index, y=d.equity["topix"], name="TOPIX",
                    line=dict(color=GREY, width=1.5, dash="dash"))
    fig.update_yaxes(type="log", title="成長倍率（対数）")
    fig.update_layout(height=460, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="gh-section">株式エクスポージャー（現金化ダイヤル）</div>',
                unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_scatter(x=d.exposure_series.index, y=d.exposure_series, name="株式比率",
                     fill="tozeroy", line=dict(color=GOOGLE_GREEN, width=1.5),
                     fillcolor="rgba(52,168,83,.12)")
    fig2.update_yaxes(range=[0, 1.05], title="株式比率", tickformat=".0%")
    fig2.update_layout(height=230, **PLOT_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown('<div class="gh-section">年次リターン：戦略 vs TOPIX</div>', unsafe_allow_html=True)
    a = d.annual
    fig3 = go.Figure()
    fig3.add_bar(x=a.index, y=a["strategy"], name="戦略", marker_color=GOOGLE_BLUE)
    fig3.add_bar(x=a.index, y=a["topix"], name="TOPIX", marker_color="#c5c9cd")
    fig3.update_layout(barmode="group", height=380, yaxis_tickformat=".0%", **PLOT_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    wins = int((a["excess"] > 0).sum())
    st.markdown(f'<span class="gh-chip">年次で対TOPIX {wins}/{len(a)} 勝</span>',
                unsafe_allow_html=True)
    st.write("")
    disp = (a * 100).round(1)
    st.dataframe(
        disp, use_container_width=True,
        column_config={
            "strategy": st.column_config.NumberColumn("戦略", format="%.1f%%"),
            "topix": st.column_config.NumberColumn("TOPIX", format="%.1f%%"),
            "excess": st.column_config.NumberColumn("超過", format="%.1f%%"),
        },
    )
