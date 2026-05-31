"""Quality/LowVol が逆効果な原因を診断（符号・外れ値・カバレッジ）。"""

from __future__ import annotations

import pandas as pd

from quant_jp.data import load, universe
from quant_jp.features import fundamentals as fnd

close = load.close_panel()
elig = universe.eligible_mask().reindex(index=close.index, columns=close.columns).fillna(False)
p = fnd.fundamental_panels(close)

d = close.index[-260]  # 直近付近の一営業日
print(f"基準日 {d.date()}, 対象銘柄数 {int(elig.loc[d].sum())}")
for k in ["EP", "BP", "OP_TA", "ROE", "EqAR"]:
    v = p[k].loc[d].where(elig.loc[d]).dropna()
    print(f"\n{k}: n={len(v)} 範囲[{v.min():.3f}, {v.max():.3f}] 中央{v.median():.3f}")
    print(f"  上位5(高い=ランク高): {[round(float(x),3) for x in v.nlargest(5)]}")
    print(f"  下位5: {[round(float(x),3) for x in v.nsmallest(5)]}")

# Quality 上位20銘柄が実際どんな企業か（ROEや収益性の値）
q = fnd.quality_score(close, elig).loc[d].dropna().sort_values(ascending=False).head(20)
print("\nQuality上位20の各指標値:")
for code in q.index[:10]:
    print(f"  {code}: OP_TA={p['OP_TA'].loc[d, code]:.3f} ROE={p['ROE'].loc[d, code]:.3f} EqAR={p['EqAR'].loc[d, code]:.3f}")
