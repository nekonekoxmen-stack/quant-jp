"""シークレット解決の一元化。

優先順位: Windows 資格情報マネージャー (keyring) -> 環境変数 -> .env(開発用)。
キーは平文でディスク/リポジトリに残さないことを優先する。アプリ側は
必ずこのモジュール経由でキーを取得し、値をログ/レポート/通知へ出力しない。
"""

from __future__ import annotations

import os
from functools import lru_cache

import keyring
from dotenv import load_dotenv

SERVICE_NAME = "quant-jp"

# .env は開発用フォールバック。存在すれば読み込む（環境変数は上書きしない）。
load_dotenv(override=False)


class SecretNotFoundError(RuntimeError):
    """要求されたシークレットがどの供給元にも存在しない。"""


def get_secret(name: str, *, required: bool = True) -> str | None:
    """名前付きシークレットを keyring -> env -> .env の順で解決する。

    Args:
        name: シークレットのキー名（例: "GEMINI_API_KEY"）。
        required: True かつ未取得なら SecretNotFoundError を送出。
    """
    # 1) Windows 資格情報マネージャー
    try:
        value = keyring.get_password(SERVICE_NAME, name)
    except Exception:
        # keyring バックエンド未構成などは黙ってフォールバック
        value = None
    if value:
        return value

    # 2) 環境変数（load_dotenv で .env も反映済み）
    value = os.environ.get(name)
    if value:
        return value

    if required:
        raise SecretNotFoundError(
            f"シークレット '{name}' が見つかりません。"
            f" `python -m quant_jp.config.secrets set {name}` で keyring に登録するか、"
            f" 環境変数 / .env に設定してください。"
        )
    return None


def set_secret(name: str, value: str) -> None:
    """シークレットを Windows 資格情報マネージャーへ保存する。"""
    keyring.set_password(SERVICE_NAME, name, value)


def delete_secret(name: str) -> None:
    """keyring からシークレットを削除する（存在しなくてもエラーにしない）。"""
    try:
        keyring.delete_password(SERVICE_NAME, name)
    except keyring.errors.PasswordDeleteError:
        pass


@lru_cache(maxsize=None)
def has_secret(name: str) -> bool:
    """シークレットが取得可能か（値は返さない）。"""
    return get_secret(name, required=False) is not None


# 既知のシークレット名。エラー文字列等のマスク対象を解決するのに使う。
_KNOWN_SECRET_NAMES = (
    "JQUANTS_API_KEY",
    "GEMINI_API_KEY",
)


def redact(text: str) -> str:
    """登録済みシークレットの値を含む文字列をマスクする。

    例外メッセージや URL にトークンが混入してログ/レポートへ漏れるのを防ぐ。
    """
    if not text:
        return text
    for name in _KNOWN_SECRET_NAMES:
        value = get_secret(name, required=False)
        if value and len(value) >= 6:
            text = text.replace(value, "***REDACTED***")
    return text


def _cli() -> None:
    """対話的に keyring へ登録するための最小 CLI。

    使い方:
      python -m quant_jp.config.secrets set GEMINI_API_KEY   # 値はプロンプトで入力
      python -m quant_jp.config.secrets check GEMINI_API_KEY
      python -m quant_jp.config.secrets delete GEMINI_API_KEY
    """
    import getpass
    import sys

    args = sys.argv[1:]
    if len(args) < 2 or args[0] not in {"set", "check", "delete"}:
        print(_cli.__doc__)
        raise SystemExit(2)

    action, name = args[0], args[1]
    if action == "set":
        # 値は端末から非表示入力。引数では受け取らない（履歴に残さない）。
        value = getpass.getpass(f"{name} の値を入力（表示されません）: ").strip()
        if not value:
            print("空の値は登録しません。")
            raise SystemExit(1)
        set_secret(name, value)
        print(f"keyring に '{name}' を登録しました。")
    elif action == "check":
        ok = get_secret(name, required=False) is not None
        print(f"{name}: {'OK（取得可能）' if ok else '未設定'}")
        raise SystemExit(0 if ok else 1)
    elif action == "delete":
        delete_secret(name)
        print(f"keyring から '{name}' を削除しました（存在した場合）。")


if __name__ == "__main__":
    _cli()
