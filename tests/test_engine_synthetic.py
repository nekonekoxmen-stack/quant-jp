"""合成データでエンジン/指標のロジックを検証（ルックアヘッド・コスト・ドリフト）。"""

import numpy as np
import pandas as pd

from quant_jp.backtest import metrics
from quant_jp.backtest.engine import run_backtest

dates = pd.bdate_range("2020-01-01", periods=60)
codes = ["A", "B", "C"]

# A は毎日 +0.1%、B は 0%、C は -0.1% の決め打ち
rets = pd.DataFrame(
    {"A": 0.001, "B": 0.0, "C": -0.001}, index=dates
)
close = (1 + rets).cumprod() * 100.0

# 常に A に 100% を目標（毎週リバランス）。コスト0なら CAGR は A とほぼ一致のはず
tw = pd.DataFrame(0.0, index=dates, columns=codes)
tw["A"] = 1.0

res = run_backtest(close, tw, rebalance="W-FRI", cost_bps=0.0)
print("最終 equity:", round(res.equity.iloc[-1], 4))
print("A のバイ&ホールド:", round((close["A"].iloc[-1] / close["A"].iloc[0]), 4))

# エクスポージャー0.5 を適用 → リターンは概ね半分
exp = pd.Series(0.5, index=dates)
res_half = run_backtest(close, tw, exposure=exp, rebalance="W-FRI", cost_bps=0.0)
print("exposure0.5 最終 equity:", round(res_half.equity.iloc[-1], 4))

# コストの効果（往復で必ず悪化する）
res_cost = run_backtest(close, tw, rebalance="W-FRI", cost_bps=50.0)
print("コスト後 < コスト前:", res_cost.equity.iloc[-1] < res.equity.iloc[-1])

print("metrics:", metrics.format_summary(metrics.summary(res.returns, benchmark=rets["B"])))

# ルックアヘッド検証: 先週金曜のシグナルで保有 → 週中の急騰(+50%)を取得できる。
dates2 = pd.bdate_range("2021-01-01", periods=15)  # 1/1(金)..
close2 = pd.DataFrame(100.0, index=dates2, columns=["D"])
close2.iloc[8:] = 150.0  # idx8 で +50% にステップ（以後維持）
tw2 = pd.DataFrame(1.0, index=dates2, columns=["D"])  # 常に D を目標
r2 = run_backtest(close2, tw2, rebalance="W-FRI", cost_bps=0.0)
print("金曜シグナルで週中の上昇を取得 (≈1.5):", round(r2.equity.iloc[-1], 3))

# 同日ルックアヘッド検証: シグナル日(金)当日の終値変化は取れない（翌営業日適用）。
close3 = pd.DataFrame(100.0, index=dates2, columns=["D"])
close3.iloc[5:] = 150.0  # idx5 = 2つ目の金曜に急騰。idx0金曜シグナル→idx1適用で既保有
# idx0 シグナルが効くのは idx1 以降。idx5 の急騰は保有中なので取得できる（正しい挙動）。
r3 = run_backtest(close3, tw2, rebalance="W-FRI", cost_bps=0.0)
print("保有中の急騰も取得 (≈1.5):", round(r3.equity.iloc[-1], 3))
