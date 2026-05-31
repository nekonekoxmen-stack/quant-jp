"""J-Quants v2 からのデータ取得バッチと Parquet データレイクへの保存。

日足・財務は年ごとのファイルに分割保存し、再開可能（既存年はスキップ）。
進捗は逐次フラッシュ、1日分の失敗は警告して継続する。

使い方:
  python -m quant_jp.data.ingest probe                       # Standard 取得範囲を検証
  python -m quant_jp.data.ingest pull 2016-05-29 2026-05-29  # 期間一括取得（再開可能）
  python -m quant_jp.data.ingest pull 2016-05-29 2026-05-29 --force  # 既存年も再取得
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from quant_jp.config.secrets import redact
from quant_jp.data.jquants_client import JQuantsClient

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def _save(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  saved {path.relative_to(ROOT)}: {len(df)} rows, {len(df.columns)} cols", flush=True)
    return path


def _year_bounds(start: str, end: str) -> list[tuple[int, str, str]]:
    """[(year, year_start_iso, year_end_iso)] を start..end の範囲でクリップして返す。"""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    for y in range(s.year, e.year + 1):
        ys = max(s, pd.Timestamp(year=y, month=1, day=1))
        ye = min(e, pd.Timestamp(year=y, month=12, day=31))
        out.append((y, ys.strftime("%Y-%m-%d"), ye.strftime("%Y-%m-%d")))
    return out


def pull(start: str, end: str, *, force: bool = False) -> None:
    """日足・上場一覧・TOPIX・財務・投資部門別売買を取得し Parquet 保存。"""
    cli = JQuantsClient()

    print("[1/5] listed_info ...", flush=True)
    _save(cli.listed_info(), DATA_DIR / "listed_info.parquet")

    print(f"[2/5] topix {start}..{end} ...", flush=True)
    _save(cli.topix(start, end), DATA_DIR / "topix.parquet")

    print(f"[3/5] investor_types {start}..{end} ...", flush=True)
    try:
        _save(cli.investor_types(start, end), DATA_DIR / "investor_types.parquet")
    except Exception as exc:  # noqa: BLE001
        print(redact(f"  skip investor_types: {type(exc).__name__}: {exc}"), flush=True)

    years = _year_bounds(start, end)

    print(f"[4/5] daily_quotes {start}..{end}（年ごと, 再開可能）...", flush=True)
    for y, ys, ye in years:
        out = DATA_DIR / "daily_quotes" / f"dq_{y}.parquet"
        if out.exists() and not force:
            print(f"  {y}: 既存のためスキップ", flush=True)
            continue
        print(f"  {y} ({ys}..{ye})", flush=True)
        df = cli.daily_quotes_range(ys, ye, progress=True)
        if not df.empty:
            _save(df, out)

    print(f"[5/5] statements {start}..{end}（年ごと, 再開可能）...", flush=True)
    for y, ys, ye in years:
        out = DATA_DIR / "statements" / f"st_{y}.parquet"
        if out.exists() and not force:
            print(f"  {y}: 既存のためスキップ", flush=True)
            continue
        print(f"  {y} ({ys}..{ye})", flush=True)
        try:
            df = cli.statements_range(ys, ye, progress=True)
            if not df.empty:
                _save(df, out)
        except Exception as exc:  # noqa: BLE001
            print(redact(f"  skip statements {y}: {type(exc).__name__}: {exc}"), flush=True)

    print("完了。", flush=True)


def update() -> None:
    """日次の増分更新（CI/運用向け）: 当年分の日足・財務を再取得し、補助データも更新。

    全期間 pull と違い当年ファイルのみ force 再取得するため軽量。過年度ファイルは
    キャッシュ済み前提（GitHub Actions のキャッシュや commit 済みデータで供給）。
    """
    import datetime as dt

    cli = JQuantsClient()
    today = dt.date.today()
    year = today.year
    ys = f"{year}-01-01"
    ye = today.isoformat()

    print("[1/5] listed_info ...", flush=True)
    _save(cli.listed_info(), DATA_DIR / "listed_info.parquet")
    print(f"[2/5] topix（全期間は不要だが安全のため当年） {ys}..{ye} ...", flush=True)
    # TOPIX は軽いので運用に必要な全期間を維持（既存とマージ）
    _merge_save(cli.topix(ys, ye), DATA_DIR / "topix.parquet", key="Date")
    print(f"[3/5] investor_types {ys}..{ye} ...", flush=True)
    try:
        _merge_save(cli.investor_types(ys, ye), DATA_DIR / "investor_types.parquet", key="PubDate")
    except Exception as exc:  # noqa: BLE001
        print(redact(f"  skip investor_types: {exc}"), flush=True)
    print(f"[4/5] daily_quotes {year}（当年再取得） ...", flush=True)
    df = cli.daily_quotes_range(ys, ye, progress=True)
    if not df.empty:
        _save(df, DATA_DIR / "daily_quotes" / f"dq_{year}.parquet")
    print(f"[5/5] statements {year}（当年再取得） ...", flush=True)
    try:
        st = cli.statements_range(ys, ye, progress=True)
        if not st.empty:
            _save(st, DATA_DIR / "statements" / f"st_{year}.parquet")
    except Exception as exc:  # noqa: BLE001
        print(redact(f"  skip statements: {exc}"), flush=True)
    print("増分更新 完了。", flush=True)


def _merge_save(df_new: pd.DataFrame, path: Path, key: str) -> None:
    """既存 Parquet と新データを key で重複排除マージして保存（全期間維持）。"""
    if path.exists() and not df_new.empty:
        old = pd.read_parquet(path)
        merged = pd.concat([old, df_new], ignore_index=True).drop_duplicates(subset=[key], keep="last")
        _save(merged.sort_values(key), path)
    elif not df_new.empty:
        _save(df_new, path)


def pull_listed_history(start: str, end: str) -> None:
    """月次の上場銘柄スナップショットを取得（point-in-time 構成＝サバイバーシップ是正用）。"""
    cli = JQuantsClient()
    months = pd.date_range(start=start, end=end, freq="MS")
    frames = []
    for i, m in enumerate(months, 1):
        d = m.strftime("%Y-%m-%d")
        print(f"  listed {d} ({i}/{len(months)})", flush=True)
        try:
            df = cli.listed_info(d)
            if not df.empty:
                df["SnapDate"] = pd.Timestamp(m)
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            print(redact(f"  skip {d}: {exc}"), flush=True)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        _save(out, DATA_DIR / "listed_history.parquet")


def probe() -> None:
    """プラン取得可否を検証し reports/plan_coverage.md にメモ化。"""
    cli = JQuantsClient()
    coverage = cli.probe_plan_coverage()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "plan_coverage.md"
    lines = ["# J-Quants プラン取得可否（実データ検証）", ""]
    for name, status in coverage.items():
        lines.append(f"- **{name}**: {status}")
    lines += [
        "",
        "## 運用メモ（v2 / Standard プラン）",
        "",
        "- API は **v2**（ベース `https://api.jquants.com/v2`）。認証は **`x-api-key` ヘッダー**（キー名 `JQUANTS_API_KEY`）。v1 のトークン方式は廃止。",
        "- `/equities/bars/daily`（日足）と `/fins/summary`（財務）は **`date` か `code` が必須**（from/to 単独は 400）。全銘柄パネルは**営業日ごとに `date=` ループ**で取得（営業日カレンダーは TOPIX から導出）。",
        "- 日足には**調整後カラムあり**: `AdjFactor, AdjO, AdjH, AdjL, AdjC, AdjVo`（分割調整に使用）。生値は `O,H,L,C`、出来高 `Vo`、売買代金 `Va`、値幅上限/下限 `UL/LL`。",
        "- 銘柄コードは **5桁**（例 トヨタ=`72030`）。",
        "- **取得可能期間: 直近約10年（ローリング）**。例: 2026-05-29 時点で `2016-05-29 〜`。範囲外を要求すると 400 でカバー範囲を返す。",
        "- 上場一覧の業種: `S17/S17Nm`(17業種), `S33/S33Nm`(33業種), `ScaleCat`(規模), `Mkt`(市場区分)。",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nメモを保存: {out}")


def _cli() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(2)
    if args[0] == "probe":
        probe()
    elif args[0] == "update":
        update()
    elif args[0] == "pull" and len(args) >= 3:
        pull(args[1], args[2], force="--force" in args)
    elif args[0] == "listed_history" and len(args) == 3:
        pull_listed_history(args[1], args[2])
    else:
        print(__doc__)
        raise SystemExit(2)


if __name__ == "__main__":
    _cli()
