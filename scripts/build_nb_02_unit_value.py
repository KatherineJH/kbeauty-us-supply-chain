"""Write notebooks/02_unit_value.ipynb (M2, pre-registration Q3 and Q4). Edit here, then:

    uv run python scripts/build_nb_02_unit_value.py
    uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_unit_value.ipynb

Interpretation text lives in scripts/nb02_interpretation.py and is written after reading the executed outputs.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "02_unit_value.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
INTERP = Path(__file__).with_name("nb02_interpretation.py")
interp = {}
if INTERP.exists():
    exec(INTERP.read_text(encoding="utf-8"), interp)
note = lambda key: md(interp.get(key, f"_(해석: 실행 결과를 확인한 뒤 작성 — {key})_"))

cells = [
    md("""# 02. 상품이 실제로 싸졌나 — 품목 구성, 품목 내 단가, 환율

미국 기록의 kg당 단가는 정책 후 해상 −16.9%, 항공 −36.0% 떨어졌다. 이것이 **상품이 싸진 것**인지 확인하려면 세 질문을 차례로 답해야 한다.

| 질문 | 방법 | 답 |
|---|---|---|
| 1. 싼 품목으로 옮겨 간 것인가, 같은 품목이 싸졌나 (M2) | 미국 HS10 기준 구성(mix)·품목 내 단가(price)·상호작용 분해 | 하락의 78~87%가 **같은 품목 안**에서 일어났다 |
| 2. 그중 환율 몫은 얼마인가 (사전 등록 Q3) | Δlog(USD/kg) = Δlog(원/kg) − Δlog(원/달러) 항등식 | 원화 약세는 하락의 **10~23%만** 설명한다 |
| 3. 같은 흐름을 한국은 어떻게 기록했나 (사전 등록 Q4) | 한국 수출 기록을 HSK10·HS6으로 분해 | 한국 기록의 kg당 금액은 **+8.0%**, 그것도 비싼 품목으로의 구성 이동이다 |

**결론**: 미국 기록의 단가 하락은 환율과 품목 구성으로 설명되지 않고, 같은 흐름의 한국 기록과 방향이 반대다. "상품이 싸졌다"보다 "미국 측 기록 가격이 달라졌다"에 가깝다(04 노트북).

비교 창은 정책 전후 같은 계절의 12개월(2024.09~2025.08 vs 2025.09~2026.08), 모든 단가는 **12개월 합계 금액 / 12개월 합계 중량**이다.
04 노트북의 월별 검정 수치(월별 평균의 전후 차이)와는 기준이 달라 숫자가 조금 다르다. 세목 단위의 분류 차이는 부록에 둔다."""),

    code("""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kbeauty import decompose, report, warehouse
from kbeauty.paths import POST_WINDOW, PRE_WINDOW, TABLES_DIR
from kbeauty.viz import annotations as an
from kbeauty.viz import notebook as nbviz
from kbeauty.viz.theme import COLORS as C

%matplotlib inline
nbviz.setup()
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
fct = warehouse.load("fct_imports_hs10_mode_monthly"); fct["month"] = pd.to_datetime(fct["month"])
kr = warehouse.load("stg_kcs__kr_exports_us"); kr["month"] = pd.to_datetime(kr["month"])
names = warehouse.load("stg_kcs__hsk10_names").set_index("hsk10")["name_ko"]
summary = warehouse.load("mart_monthly_summary"); summary["month"] = pd.to_datetime(summary["month"])
krw = warehouse.load("stg_fed_board__fx_monthly").query("currency == 'KRW'")
krw = krw.assign(month=pd.to_datetime(krw["month"])).set_index("month")["lcu_per_usd"]
in_window = lambda d, w: d[d["month"].between(*w)]
print("US HS10 lines:", fct.hs10.nunique(), "| Korean HSK10 lines:", kr.hsk10.nunique())"""),

    md("""## 1. 미국 기록 (M2): 하락은 품목 안에서 일어났다"""),

    code("""us_mode = decompose.decompose_by_mode(fct, PRE_WINDOW, POST_WINDOW).set_index("mode")
sea_air = fct[fct["mode"] != "other"].groupby(["month", "hs10"], as_index=False)[["value_usd", "weight_kg"]].sum()
us_mode.loc["sea+air"] = pd.Series(decompose.window_decomposition(sea_air, "hs10", "value_usd", "weight_kg", PRE_WINDOW, POST_WINDOW))
us_mode[["uv_base", "uv_comp", "change", "change_pct", "mix", "price", "interaction", "mix_share_of_change", "price_share_of_change"]]"""),

    note("us"),

    md("""## 2. 환율 (Q3): 원화 약세가 달러 단가를 얼마나 끌어내렸나

수출업체가 원화 가격을 그대로 두어도, 원화가 약해지면 달러로 환산한 단가는 내려간다. 12개월 합계 기준으로
달러 단가 변화를 '원화 단가 변화'와 '금액 가중 평균 환율 변화'로 정확히 나눈다."""),

    code("""fig, ax = nbviz.figure(title="원화는 정책 후 12개월 동안 약해졌다", subtitle="원/달러 월평균. 음영 = 정책 전·후 12개월 비교 창")
k = krw.loc["2024-01-01":"2026-08-01"]
for (a, b), c, name in ((PRE_WINDOW, C["grid"], "정책 전 12개월"), (POST_WINDOW, C["primary"], "정책 후 12개월")):
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0), color=c, alpha=0.3, lw=0)
    an.note(ax, pd.Timestamp(a), 1.0, name, coords=("data", "axes fraction"), va="top", xytext=(4, -4))
ax.plot(k.index, k, color=C["ink"], lw=2)
ax.set_ylabel("원 / 달러")
plt.show()
fx_tbl = pd.read_csv(TABLES_DIR / "fx_decomposition.csv").set_index("series")
fx_tbl[["usd_per_kg_pre", "usd_per_kg_post", "rate_pre", "rate_post", "dlog_usd_per_kg", "dlog_lcu_per_kg", "dlog_rate", "fx_share_of_change"]]"""),

    md("""아래 셀은 위 표(`run_all` 결과)를 노트북에서 다시 계산해 같은지 확인한다."""),

    code("""ss = summary.set_index("month")
check = decompose.fx_decomposition(ss.air_value_usd, ss.air_weight_kg, krw.reindex(ss.index), PRE_WINDOW, POST_WINDOW)
assert np.isclose(check["dlog_usd_per_kg"], fx_tbl.loc["US imports, air", "dlog_usd_per_kg"])
assert np.isclose(check["dlog_usd_per_kg"], check["dlog_lcu_per_kg"] - check["dlog_rate"])  # identity
print("identity holds; matches reports/tables/fx_decomposition.csv")"""),

    note("fx"),

    md("""## 3. 한국 기록 (Q4): 같은 흐름을 한국은 어떻게 기록했나

한국 관세청 기록은 HSK 10단위로 미국보다 일부 더 세밀하다(예: 미국 3304.99 한 세목 ↔ 한국 4개 세목).
한국 측 kg당 금액 변화를 세목 단위로 분해하고, 같은 방법을 HS6 단위와 미국 측에도 적용해 비교한다."""),

    code("""gran = pd.read_csv(TABLES_DIR / "granularity_decomposition.csv").set_index("series")
gran[["uv_base", "uv_comp", "change", "change_pct", "mix", "price", "interaction", "mix_share_of_change", "price_share_of_change"]]"""),

    code("""def item_table(df, item, value, weight):
    p0 = in_window(df, PRE_WINDOW).groupby(item)[[value, weight]].sum()
    p1 = in_window(df, POST_WINDOW).groupby(item)[[value, weight]].sum()
    t = pd.DataFrame({"kg_share_pre": p0[weight] / p0[weight].sum(), "kg_share_post": p1[weight] / p1[weight].sum(),
                      "usd_per_kg_pre": p0[value] / p0[weight], "usd_per_kg_post": p1[value] / p1[weight],
                      "value_post_musd": p1[value] / 1e6})
    t["share_change_pp"] = (t.kg_share_post - t.kg_share_pre) * 100
    t["mix_contribution"] = (t.kg_share_post - t.kg_share_pre) * t.usd_per_kg_pre
    return t.sort_values("value_post_musd", ascending=False)

kr_items = item_table(kr, "hsk10", "export_usd", "export_kg")
kr_items.insert(0, "name_ko", names.reindex(kr_items.index).values)
kr_items.head(12)"""),

    note("kr_items"),

    note("conclusion"),

    md("""---

## 부록: 두 나라 세목의 분류 차이와 일치 확인"""),

    md("""### A1. 3304.99 안쪽: 미국이 한 세목으로 묶은 것을 한국은 넷으로 나눈다"""),

    code("""sub = kr_items.loc[kr_items.index.str.startswith("330499")]
sub[["name_ko", "kg_share_pre", "kg_share_post", "usd_per_kg_pre", "usd_per_kg_post", "value_post_musd", "mix_contribution"]]"""),

    note("sub"),

    md("""### A2. 마스크 팩은 어디로 갔나

한국 HSK에는 '마스크 팩'(3307.90-4000) 세목이 따로 있다. 04 노트북에서 330790(기타 화장품 조제품)은 미국 기록이 한국의 34%에 불과했다.
마스크 팩을 미국이 다른 세번(예: 3304.99)으로 분류한다면 HS6 단위 비교와 구성 분해가 모두 영향을 받는다."""),

    code("""mask = kr[kr.hsk10 == "3307904000"].set_index("month")[["export_usd", "export_kg"]]
k_3307 = kr[kr.hs6 == "330790"].groupby("month")["export_usd"].sum()
us_by_hs6 = fct[fct["mode"] != "other"].assign(hs6=lambda d: d.hs10.str[:6]).groupby(["month", "hs6"])["value_usd"].sum().unstack()
mask_tbl = pd.DataFrame({
    "KR 마스크 팩 (3307.90-4000)": [in_window(kr.query("hsk10 == '3307904000'"), w).export_usd.sum() / 1e6 for w in (PRE_WINDOW, POST_WINDOW)],
    "KR 330790 전체": [in_window(kr.query("hs6 == '330790'"), w).export_usd.sum() / 1e6 for w in (PRE_WINDOW, POST_WINDOW)],
    "US 330790 (해상+항공)": [us_by_hs6.loc[w[0]:w[1], "330790"].sum() / 1e6 for w in (PRE_WINDOW, POST_WINDOW)],
    "KR 330499 전체": [in_window(kr.query("hs6 == '330499'"), w).export_usd.sum() / 1e6 for w in (PRE_WINDOW, POST_WINDOW)],
    "US 330499 (해상+항공)": [us_by_hs6.loc[w[0]:w[1], "330499"].sum() / 1e6 for w in (PRE_WINDOW, POST_WINDOW)],
}, index=["정책 전 12개월 (백만 달러)", "정책 후 12개월 (백만 달러)"]).T
mask_tbl["mask_share_of_kr_330790"] = np.nan
mask_tbl.loc["KR 마스크 팩 (3307.90-4000)", "mask_share_of_kr_330790"] = (mask_tbl.iloc[0, 1] / mask_tbl.iloc[1, 1])
mask_tbl"""),

    note("mask"),

    md("""### A3. 일치 확인

미국 측 운송수단별 분해가 `scripts/run_all.py`의 표(`reports/tables/unit_value_decomposition.csv`)와 같은지 확인한다."""),

    code("""saved = pd.read_csv(TABLES_DIR / "unit_value_decomposition.csv").set_index("mode")
for mode in ("vessel", "air"):
    for col in ("mix", "price", "interaction"):
        assert np.isclose(saved.loc[mode, col], us_mode.loc[mode, col], atol=1e-12)
print("notebook decomposition matches reports/tables/unit_value_decomposition.csv")"""),

]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
nbf.write(nb, OUT)
print(f"wrote {OUT}")
