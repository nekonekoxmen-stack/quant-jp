"""Gemini API による調査レイヤー。

モデルは GEMINI_MODEL（gemini-3.5-flash）で固定。API キーは config.secrets
経由で解決し、平文では保持しない。調査結果は reports/research/ に Markdown 保存。

使い方:
  python -m quant_jp.research.gemini_research "日本株のクロスセクション・モメンタムの有効性に関する近年の知見"
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

from quant_jp.config.secrets import get_secret

# ユーザー指定により固定。変更しないこと。
GEMINI_MODEL = "gemini-3.5-flash"

RESEARCH_DIR = Path(__file__).resolve().parents[3] / "reports" / "research"


def _client():  # noqa: ANN202 - 外部ライブラリ型
    from google import genai  # 遅延 import

    api_key = get_secret("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def research(prompt: str, *, save: bool = True) -> str:
    """調査プロンプトを gemini-3.5-flash で実行し、本文テキストを返す。"""
    client = _client()
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = response.text or ""
    if save:
        _save_markdown(prompt, text)
    return text


def _save_markdown(prompt: str, text: str) -> Path:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^0-9A-Za-z぀-ヿ一-鿿]+", "_", prompt)[:40].strip("_")
    out = RESEARCH_DIR / f"{ts}_{slug or 'research'}.md"
    body = (
        f"# 調査メモ\n\n"
        f"- 日時: {dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"- モデル: {GEMINI_MODEL}\n\n"
        f"## プロンプト\n\n{prompt}\n\n## 回答\n\n{text}\n"
    )
    out.write_text(body, encoding="utf-8")
    return out


def _cli() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        raise SystemExit(2)
    prompt = " ".join(args)
    text = research(prompt)
    print(text)


if __name__ == "__main__":
    _cli()
