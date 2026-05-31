# 自動ペーパートレード・ダッシュボードの公開手順

GitHub Actions が**平日の場引け後（JST 16:30）に自動で**データ更新→ペーパートレード→
ダッシュボード生成を行い、GitHub Pages に公開します。PC を起動していなくても動きます。

以下はあなたが一度だけ行う設定です（各数分）。`gh`（GitHub CLI）導入済み・ローカル commit 済み。

> 注意: PowerShell を開き直すと `gh` がそのまま使えます。使えない場合はフルパス
> `& "$env:LOCALAPPDATA\Microsoft\WinGet\Links\gh.exe"` を `gh` の代わりに使ってください。

## 1. GitHub にログイン

```powershell
gh auth login
```
- `GitHub.com` → `HTTPS` → `Login with a web browser` を選択
- 表示される8桁コードを控え、Enter でブラウザが開く → コード入力して認証

## 2. リポジトリ作成＆プッシュ（プライベート推奨）

```powershell
cd C:\Users\Ryo\Claude\quant-jp
gh repo create quant-jp --private --source=. --remote=origin --push
```
これでコード一式があなたの GitHub に上がります（データ・APIキーは .gitignore で除外済み）。

## 3. APIキーを GitHub Secrets に登録（あなたの手で・暗号化保管）

キーは画面に表示されません。プロンプトに貼り付けて Enter。

```powershell
gh secret set JQUANTS_API_KEY
gh secret set GEMINI_API_KEY   # 任意（調査機能を使う場合）
```
確認: `gh secret list`

## 4. GitHub Pages を有効化（ソース = GitHub Actions）

```powershell
gh api -X POST repos/:owner/quant-jp/pages -f build_type=workflow
```
うまくいかない場合はブラウザで: リポジトリ → Settings → Pages →
「Build and deployment」の Source を **GitHub Actions** に設定。

## 5. 初回実行（手動キック）

```powershell
gh workflow run daily-paper-trade
```
- 初回はデータ10年分を取得するため**数十分**かかります（以降はキャッシュで数分）。
- 進捗: `gh run watch` または リポジトリの **Actions** タブ
- 完了後、公開URL: `https://<あなたのユーザー名>.github.io/quant-jp/`
  （URLは `gh browse --settings` の Pages 欄でも確認可）

## 以降の運用

- **何もしなくて OK**。平日 16:30 JST に自動更新され、上記URLが毎日新しくなります。
- スマホのホーム画面にURLを追加すれば、毎日の推奨ポートフォリオ・仮想口座成績をすぐ確認できます。
- ペーパー台帳 `reports/paper_ledger.csv` は毎日 commit され履歴が残ります。

## ペーパートレードの設定（変更したい場合）

`src/quant_jp/ops/paper_trade.py` 冒頭:
- `START_DATE = "2026-06-01"` … 仮想運用の開始日
- `INIT_CAPITAL = 1_500_000` … 初期資金（150万円）

変更後は commit→push すれば次回実行から反映されます。

## ローカルでの確認（任意）

```powershell
uv run python -m quant_jp.ops.paper_trade        # 台帳更新＋サマリ
uv run python -m quant_jp.dashboard.build_static # site/index.html 生成
```
