"""依存・自作モジュールの import と J-Quants/Gemini の API シグネチャ確認。"""

from quant_jp.config import secrets  # noqa: F401
from quant_jp.data import ingest, jquants_client  # noqa: F401
from quant_jp.research import gemini_research

print("quant_jp modules import OK")
print("GEMINI_MODEL =", gemini_research.GEMINI_MODEL)

from google import genai  # noqa: E402

print("google.genai OK (Client:", hasattr(genai, "Client"), ")")

import jquantsapi  # noqa: E402

methods = {x for x in dir(jquantsapi.Client) if not x.startswith("_")}
need = [
    "get_listed_info",
    "get_price_range",
    "get_indices_topix",
    "get_statements_range",
    "get_markets_trades_spec",
    "get_prices_daily_quotes",
]
for n in need:
    print(f"  {n}: {'OK' if n in methods else 'MISSING'}")
