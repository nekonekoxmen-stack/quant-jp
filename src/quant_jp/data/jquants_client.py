"""J-Quants API v2 クライアント（requests 直叩き）。

認証は v2 の API キー方式（`x-api-key` ヘッダー）。キーは config.secrets 経由
（JQUANTS_API_KEY）で解決し、平文では保持しない。レスポンスは pagination_key
で分割されるため全ページを結合して返す。

参考: ベース https://api.jquants.com/v2 / 認証 x-api-key / ページング pagination_key
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import requests

from quant_jp.config.secrets import get_secret, redact

BASE_URL = "https://api.jquants.com/v2"

# v2 エンドポイント
EP_LISTED = "/equities/master"
EP_DAILY = "/equities/bars/daily"
EP_TOPIX = "/indices/bars/daily/topix"
EP_STATEMENTS = "/fins/summary"
EP_INVESTOR = "/equities/investor-types"


class JQuantsError(RuntimeError):
    pass


class JQuantsClient:
    def __init__(self, *, timeout: int = 60, max_retries: int = 5) -> None:
        self._api_key = get_secret("JQUANTS_API_KEY")
        self._timeout = timeout
        self._max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"x-api-key": self._api_key})

    # ---- 低レベル ---------------------------------------------------------

    def _request(self, path: str, params: dict[str, str]) -> requests.Response:
        """1回分の GET。429/5xx/通信エラーは指数バックオフでリトライ。"""
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(BASE_URL + path, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                last_exc = exc
            else:
                # レート制限/サーバ側一時エラーはリトライ、それ以外の 4xx は即時失敗
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_exc = JQuantsError(
                        redact(f"{path} -> HTTP {resp.status_code} body={resp.text[:200]}")
                    )
                else:
                    return resp
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        raise JQuantsError(redact(f"{path} -> リトライ上限。最後のエラー: {last_exc}"))

    def _get_pages(self, path: str, params: dict[str, str] | None = None) -> pd.DataFrame:
        """1エンドポイントを pagination_key 完了まで取得し DataFrame で返す。"""
        params = {k: v for k, v in (params or {}).items() if v}
        rows: list[dict] = []
        page_key: str | None = None
        while True:
            q = dict(params)
            if page_key:
                q["pagination_key"] = page_key
            resp = self._request(path, q)
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                raise JQuantsError(redact(f"{path} -> {exc} body={resp.text}")) from None
            body = resp.json()
            data_key = _data_key(body)
            if data_key is not None:
                rows.extend(body[data_key])
            page_key = body.get("pagination_key")
            if not page_key:
                break
        return pd.DataFrame(rows)

    # ---- 取得系 -----------------------------------------------------------

    def listed_info(self, date: str | None = None) -> pd.DataFrame:
        """上場銘柄一覧（ユニバースの母集合）。date は YYYYMMDD。"""
        return self._get_pages(EP_LISTED, {"date": _compact(date) if date else ""})

    def topix(self, start: str, end: str) -> pd.DataFrame:
        """TOPIX 指数（ベンチマーク）。from/to 受付。"""
        return self._get_pages(EP_TOPIX, {"from": _compact(start), "to": _compact(end)})

    def investor_types(self, start: str, end: str) -> pd.DataFrame:
        """投資部門別売買（外国人フロー等、レジーム特徴量）。from/to 受付。"""
        return self._get_pages(EP_INVESTOR, {"from": _compact(start), "to": _compact(end)})

    def trading_days(self, start: str, end: str) -> list[str]:
        """営業日カレンダー（YYYYMMDD のリスト）を TOPIX から導出。"""
        topix = self.topix(start, end)
        if topix.empty or "Date" not in topix.columns:
            return []
        return [str(d).replace("-", "") for d in sorted(topix["Date"].unique())]

    # /equities/bars/daily と /fins/summary は date か code が必須。
    def daily_quotes_by_date(self, date: str) -> pd.DataFrame:
        """指定営業日の全銘柄日足四本値。date は YYYYMMDD。"""
        return self._get_pages(EP_DAILY, {"date": _compact(date)})

    def daily_quotes_by_code(self, code: str, start: str, end: str) -> pd.DataFrame:
        """単一銘柄の日足四本値（期間）。"""
        return self._get_pages(
            EP_DAILY, {"code": code, "from": _compact(start), "to": _compact(end)}
        )

    def daily_quotes_range(self, start: str, end: str, *, progress: bool = False) -> pd.DataFrame:
        """期間の全銘柄日足を、営業日ごとに取得して結合。1日分の失敗は警告して継続。"""
        days = self.trading_days(start, end)
        frames: list[pd.DataFrame] = []
        for i, d in enumerate(days, 1):
            if progress:
                print(f"    daily_quotes {d} ({i}/{len(days)})", end="\r", flush=True)
            try:
                frames.append(self.daily_quotes_by_date(d))
            except JQuantsError as exc:
                print(redact(f"\n    WARN daily_quotes {d}: {exc}"), flush=True)
        if progress and days:
            print(flush=True)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def statements_by_date(self, date: str) -> pd.DataFrame:
        """指定日に開示された財務サマリ。date は YYYYMMDD。"""
        return self._get_pages(EP_STATEMENTS, {"date": _compact(date)})

    def statements_range(self, start: str, end: str, *, progress: bool = False) -> pd.DataFrame:
        """期間の財務サマリを、営業日ごとに取得して結合（DisclosedDate ベース）。"""
        days = self.trading_days(start, end)
        frames: list[pd.DataFrame] = []
        for i, d in enumerate(days, 1):
            if progress:
                print(f"    statements {d} ({i}/{len(days)})", end="\r", flush=True)
            try:
                df = self.statements_by_date(d)
            except JQuantsError as exc:
                print(redact(f"\n    WARN statements {d}: {exc}"), flush=True)
                continue
            if not df.empty:
                frames.append(df)
        if progress and days:
            print(flush=True)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ---- 検証 -------------------------------------------------------------

    def probe_plan_coverage(self, lookback_days: int = 30) -> dict[str, str]:
        """各エンドポイントの取得可否を実データで確認し要約する（Day1 検証）。"""
        end = dt.date.today()
        start = end - dt.timedelta(days=lookback_days)
        s, e = start.isoformat(), end.isoformat()
        days = self.trading_days(s, e)
        last_day = days[-1] if days else end.strftime("%Y%m%d")
        checks = {
            "listed_info": lambda: self.listed_info(),
            "daily_quotes": lambda: self.daily_quotes_by_date(last_day),
            "topix": lambda: self.topix(s, e),
            "statements": lambda: self.statements_by_date(last_day),
            "investor_types": lambda: self.investor_types(s, e),
        }
        result: dict[str, str] = {}
        for name, fn in checks.items():
            try:
                df = fn()
                cols = ", ".join(map(str, df.columns[:10]))
                latest = ""
                for c in ("Date", "DisclosedDate", "PublishedDate"):
                    if c in df.columns and len(df):
                        latest = f" / latest {c}={df[c].max()}"
                        break
                result[name] = f"OK rows={len(df)} cols=[{cols}]{latest}"
            except Exception as exc:  # noqa: BLE001 - 検証目的で全例外を記録
                result[name] = redact(f"NG {type(exc).__name__}: {exc}")
        return result


def _data_key(body: dict) -> str | None:
    """レスポンス本文から「データ配列」のキー名を検出（pagination_key 以外の list 値）。"""
    for k, v in body.items():
        if k != "pagination_key" and isinstance(v, list):
            return k
    return None


def _compact(date_str: str) -> str:
    """'YYYY-MM-DD' または 'YYYYMMDD' を 'YYYYMMDD' に正規化。"""
    return date_str.replace("-", "")
