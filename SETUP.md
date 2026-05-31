# quant-jp セットアップ

日本株 分析・売買シグナル生成システム（J-Quants / スイング中心 / レジーム現金化オーバーレイ）。

## 1. 環境構築（uv + Python 3.12）

uv は winget で導入済み（`C:\Users\Ryo\AppData\Local\Microsoft\WinGet\Links\uv.exe`）。
シェルを再起動すれば `uv` がそのまま使えます。

```powershell
cd C:\Users\Ryo\Claude\quant-jp
uv sync            # Python 3.12 を取得し .venv を作成、依存をインストール
```

## 2. シークレット登録（最重要・本人が実施）

API キーは **Windows 資格情報マネージャー（keyring）** に保存します。平文ファイルやリポジトリには残しません。
値はプロンプトで非表示入力され、コマンド履歴にも残りません。

```powershell
# J-Quants（リフレッシュトークン方式の例）
uv run python -m quant_jp.config.secrets set JQUANTS_REFRESH_TOKEN

# もしくは メール/パスワード方式
uv run python -m quant_jp.config.secrets set JQUANTS_MAIL_ADDRESS
uv run python -m quant_jp.config.secrets set JQUANTS_PASSWORD

# Gemini（調査レイヤー、モデルは gemini-3.5-flash 固定）
uv run python -m quant_jp.config.secrets set GEMINI_API_KEY

# 確認
uv run python -m quant_jp.config.secrets check GEMINI_API_KEY
```

> 開発用フォールバックとして `.env`（`.env.example` をコピー）も利用可能ですが、平文保存になるため非推奨です。`.env` は `.gitignore` 済み。

## 3. データ取得と検証

```powershell
# Standard プランで実際に取得できる範囲を検証（reports/plan_coverage.md に保存）
uv run python -m quant_jp.data.ingest probe

# 期間一括取得（Parquet を data/ に保存）
uv run python -m quant_jp.data.ingest pull 2020-01-01 2026-05-29
```

## 4. バックテストと売買シグナル

```powershell
# 上場履歴（月次, サバイバーシップ是正用）を取得（初回のみ）
uv run python -m quant_jp.data.ingest listed_history 2016-05-01 2026-05-01

# 対TOPIXバックテスト（reports/baseline_result.md, baseline_equity.png）
uv run python -m quant_jp.backtest.run_baseline

# 今日の推奨ポートフォリオ（reports/daily_signal.md）
uv run python -m quant_jp.ops.daily_report --capital 3000000

# パラメータ探索・頑健性チェック
uv run python scripts\sweep_overlay.py
uv run python scripts\robustness.py
```

戦略概要: 東証主力市場（point-in-time）の流動性上位から、**モメンタム＋バリュー＋クオリティ**の
合成スコア上位15銘柄を逆ボラ加重で月次保有。**ボラ目標0.15**で弱気局面は自動的に現金化。
実績(2017-2026, コスト後): CAGR 14.0% / Sharpe 0.88 / MaxDD -28.9%（TOPIX 11.0%/0.66/-35.3%）。

## 5. ダッシュボード（Streamlit）

```powershell
uv run streamlit run src/quant_jp/dashboard/app.py
# ブラウザで http://localhost:8501
# スマホ等から同一LANで見る場合:
uv run streamlit run src/quant_jp/dashboard/app.py --server.address 0.0.0.0
```

タブ構成: ①今日の推奨ポートフォリオ（1株単位の株数）②資産曲線＋現金化エクスポージャー
③年次の対TOPIX比較。サイドバーで資金・銘柄数・ボラ目標・コスト・リバランスを変更可能。

## 6. 調査（Gemini）

```powershell
uv run python -m quant_jp.research.gemini_research "日本株のクロスセクション・モメンタムに関する近年の知見"
# 結果は reports/research/ に Markdown 保存
```
